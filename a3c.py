from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import WANGPolicy
#from i2a import SimplePolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

collect_seed_transition_probs = []

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)
    
    features = rollout.features[0]
    batch_pa =  np.append(np.asarray([[0,0], [0,0]]), batch_a[:-2], axis=0)
    batch_pr = np.append(np.asarray([0,0]), rewards[:-2], axis = 0)
    batch_pr = np.reshape(batch_pr, (-1,1))
    
    #print(np.concatenate((batch_a, batch_pa), axis=1))
    #print(np.concatenate((np.reshape(rewards, (-1,1)), batch_pr), axis=1))
    #print(np.concatenate((batch_si, batch_a, np.reshape(rewards, (-1,1))), axis = 1))
    
    return Batch(batch_si, batch_a, batch_pa, batch_adv, batch_r, batch_pr, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "pa", "adv", "r", "pr", "terminal", "features"])

class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, policy, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)



def env_runner(env, policy, num_local_steps, summary_writer, render):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state = env.reset()
    last_pa = [0, 0]
    last_pr = [0]
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, last_pa, last_pr, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]
            #features = policy.get_initial_features()            
            
            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            last_pa = action
            last_pr = [reward]
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            #timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            timestep_limit = 3e8
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                    last_pa = [0, 0]
                    last_pr = [0]                    
                last_features = policy.get_initial_features()    
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, last_pa, last_pr, *last_features)          
            
        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class A3C(object):
    def __init__(self, env, task, visualise):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = WANGPolicy(env.observation_space.n, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                params = tf.trainable_variables()
                name_scope = tf.contrib.framework.get_name_scope()

                # Used if we are loading in a scope different than what we saved in.
                def fix_tf_name(name, name_scope=None):
                    if name_scope is not None:
                        name = name[len(name_scope) + 1:]
                    return name.split(':')[0]

                if len(name_scope) != 0:
                    params = {fix_tf_name(v.name, name_scope): v for v in params}
                else:
                    params = {fix_tf_name(v.name): v for v in params}                
                    
                self.saver = tf.train.Saver(params, max_to_keep=15)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = WANGPolicy(env.observation_space.n, env.action_space.n)
                pi.global_step = self.global_step
            
            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.05 * vf_loss - entropy * 0.05

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, 200, visualise)

            grads = tf.gradients(self.loss, pi.var_list)

            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                #tf.summary.image("model/state", pi.state)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                #tf.image_summary("model/state", pi.state)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.merge_all_summaries()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            #print(pi.var_list)
            #print(self.network.var_list)
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.RMSPropOptimizer(learning_rate=7e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0
            
    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.9, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        #should_save_plot = self.task==0 and self.local_steps % 100 == 0
        should_save_plot = self.local_steps % 100 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]
            
        if should_save_plot:
            self.plot(self.local_steps, True)

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.local_network.pa: batch.pa,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.pr: batch.pr,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],            
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        should_save_ac = fetched[-1] % 100 == 0
        if should_save_ac:
            if not os.path.exists('weight'):
                os.makedirs('weight')
            self.saver.save(sess, 'weight/a3c_'+str(fetched[-1])+'.ckpt')           
        
        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

    def save(self, sess, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(sess, path + '/' + name)

    def load(self, sess, full_path):
        self.saver.restore(sess, full_path)
        
    def plot(self, episode_count, train):
        
        plot_path = './plots'
        fig, ax = plt.subplots()
        x = np.arange(2)
        ax.set_ylim([0.0, 1.0])
        ax.set_ylabel('Stay Probability')

        stay_probs = self.env.env.env.env.env.stayProb()

        common = [stay_probs[0,0,0],stay_probs[1,0,0]]
        uncommon = [stay_probs[0,1,0],stay_probs[1,1,0]]

        collect_seed_transition_probs.append([common,uncommon])

        ax.set_xticks([1.3,3.3])
        ax.set_xticklabels(['Last trial rewarded', 'Last trial not rewarded'])

        c = plt.bar([1,3],  common, color='b', width=0.5)
        uc = plt.bar([1.8,3.8], uncommon, color='r', width=0.5)
        ax.legend( (c[0], uc[0]), ('common', 'uncommon') )
        if train:
            plt.savefig(plot_path +"/"+ 'train_' + str(episode_count) + ".png")
        else:
            plt.savefig(plot_path +"/"+ 'test_' + str(episode_count) + ".png")
        self.env.transition_count = np.zeros((2,2,2))        
        
def get_a3c(env,task,visualise):
    actor_critic = A3C(env,task,visualise)
    return actor_critic        
