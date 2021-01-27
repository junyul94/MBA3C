#This code is written by @sracaniere from DeepMind
#https://github.com/sracaniere

import numpy as np
import math

# encoding of the higher stages
S_1 = 0
S_2 = 1
S_3 = 2
nb_states = 3

class SwitchTwostep(object):
    def __init__(self, frame_cap=200):
        self.frame_cap = frame_cap
        self.state = S_1
        # defines what is the stage with the highest expected reward. Initially random
        self.highest_reward_second_stage = np.random.choice([S_2,S_3])
        self.num_actions = 2
        self.episode_step = 0        
        self.start()    
        
        self.timestep = 0
        self.reward = 0.
        self.pcontinue = 1
        self.timer = 0
        self.step_reward = 0
        self.init_state = np.random.randint(2)
        self.mode = self.episode_step // 10 % 2 # 0은 MB가 유리, 1은 MF가 유리

        # initialization of plotting variables
        common_prob_MB = 0.8
        self.transitions_MB = np.array([
            [common_prob_MB, 1-common_prob_MB],
            [1-common_prob_MB, common_prob_MB]
        ])    
        
        common_prob_MF = 1.
        self.transitions_MF = np.array([
            [common_prob_MF, 1-common_prob_MF],
            [1-common_prob_MF, common_prob_MF]
        ])     
        
        if(self.mode == 0):
            self.transitions = self.transitions_MB
        else:         
            self.transitions = self.transitions_MF
            
        self.transition_count = np.zeros((2,2,2))

        self.last_action = None
        self.last_state = None    

    def start(self):
        """Starts a new episode."""        
        # for the two-step task plots
        self.last_is_common = None
        self.last_is_rewarded = None
        self.last_action = None
        self.last_state = None

        self.mode = np.random.randint(2) # 0은 MB가 유리, 1은 MF가 유리
        
        # come back to S_1 at the end of an episode
        self.state = S_1               

        self.episode_step +=1
        self.mode = self.episode_step // 10 % 2
        self.timestep = 0
        self.reward = 0
        self.pcontinue = 1
        self.init_state = np.random.randint(2)
        return self._state2onehot(self.state), self.reward, self.pcontinue

    def step(self, action):
        """Advances environment one time-step following the given action."""
        self.timestep += 1
        self.pcontinue = 1
        self.reward = self.step_reward
        self.timer += 1
        self.last_state = self.state        

        # get next stage
        if (self.state == S_1):
            if(self.mode == 0):
                self.transitions = self.transitions_MB
                self.possible_switch()
                
            
            # get reward
            self.reward = 0
            # update stage
            self.state = S_2 if (np.random.uniform() < self.transitions[action][0]) else S_3
            # keep track of stay probability after first action
            if (self.last_action != None):    
                self.updateStateProb(action)
            self.last_action = action
            # book-keeping for plotting
            self.last_is_common = self.isCommon(action,self.state-1)
            
            if(self.mode == 1):
                self.transitions = self.transitions_MF
                if (self.last_is_rewarded == 1):
                    self.switch()
            
            # get probability of reward in stage
            r_prob = 0.9 if (self.highest_reward_second_stage == self.state) else 0.1
            # get reward
            self.reward = 1 if np.random.uniform() < r_prob else 0            
            
            # book-keeping for plotting
            self.last_is_rewarded = self.reward            
            
            self.state = S_1

        # new state after the decision
        new_state = self.get_state()
        if self.timestep >= 200: 
            done = True
        else: 
            done = False
            
        self.pcontinue = int(not done)
        
        return new_state, self.reward, self.pcontinue, self.timestep

    def observation(self, agent_id=0):
        return (self.reward,
                self.pcontinue,
                self._state2onehot(self.state))

    def _state2onehot(self, x):
        x_onehot = np.zeros(3)
        x_onehot[x] = 1
        return x_onehot
    
    def _action2onehot(self, x):
        x_onehot = np.zeros(2)
        x_onehot[x] = 1
        return x_onehot
    
    def get_state(self):
          one_hot_array = np.zeros(nb_states)
          one_hot_array[self.state] = 1
          return one_hot_array    
    
    def possible_switch(self):
        if (np.random.uniform() < 0.025):
            # switches which of S_2 or S_3 has expected reward of 0.9
            self.highest_reward_second_stage = S_2 if (self.highest_reward_second_stage == S_3) else S_3    
            
    def switch(self):
        if (np.random.uniform() < 0.8):
            # switches which of S_2 or S_3 has expected reward of 0.9
            self.highest_reward_second_stage = S_2 if (self.highest_reward_second_stage == S_3) else S_3            
        
    def get_rprobs(self):
        """
        probability of reward of states S_2 and S_3, in the form [[p, 1-p], [1-p, p]]
        """
        if (self.highest_reward_second_stage == S_2):
            r_prob = 0.9
        else:
            r_prob = 0.1

        rewards = np.array([
            [r_prob, 1-r_prob],
            [1-r_prob, r_prob]
        ])
        return rewards
            
    def isCommon(self,action,state):
        if self.transitions[action][state] >= 1/2:
            return True
        return False
        
    def updateStateProb(self,action):
        if self.last_is_rewarded: #R
            if self.last_is_common: #C
                if self.last_action == action: #Rep
                    self.transition_count[0,0,0] += 1
                else: #URep
                    self.transition_count[0,0,1] += 1
            else: #UC
                if self.last_action == action: #Rep
                    self.transition_count[0,1,0] += 1
                else: #URep
                    self.transition_count[0,1,1] += 1
        else: #UR
            if self.last_is_common:
                if self.last_action == action:
                    self.transition_count[1,0,0] += 1
                else:
                    self.transition_count[1,0,1] += 1
            else:
                if self.last_action == action:
                    self.transition_count[1,1,0] += 1
                else:
                    self.transition_count[1,1,1] += 1                    
        
    def stayProb(self):
        print(self.transition_count)
        row_sums = self.transition_count.sum(axis=-1)
        stay_prob = self.transition_count / row_sums[:,:,np.newaxis] 
       
        return stay_prob        
    
    def is_MF(self):
        return self.mode