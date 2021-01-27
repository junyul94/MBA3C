# From @higgsfield
# https://github.com/higgsfield/Imagination-Augmented-Agents

import gym
import numpy as np
from gym import spaces
from gym.utils import closer
#from common.deepmind import PillEater, observation_as_rgb
from common.TwoStep import SwitchTwostep

env_closer = closer.Closer()

class Simple:
    def __init__(self, mode, frame_cap):
        self.frame_cap = frame_cap

        self.env = SwitchTwostep(frame_cap=frame_cap)

        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self.metadata = {}
        self.spec = {}
        self.reward_range = (np.inf, np.inf)
        self._configured = False
        
    def _close(self):
        pass
        
    def _configure(self):
        pass        

    def step(self, action):
        self.env.step(action)
        env_reward, env_pcontinue, env_frame= self.env.observation()
        self.done = env_pcontinue != 1
        #env_frame = env_frame.transpose(2, 0, 1)
        return env_frame, env_reward, self.done, {}

    def reset(self):
        image, _, _ = self.env.start()
        #image = observation_as_rgb(image)
        self.done = False
        #image = image.transpose(2, 0, 1)
        return image
    
    def unwrapped(self):
        return self
    
    def configure(self, *args, **kwargs):
        self._configured = True
        
        try:
            self._configure(*args, **kwargs)
        except TypeError as e:
            # It can be confusing if you have the wrong environment
            # and try calling with unsupported arguments, since your
            # stack trace will only show core.py.
            if self.spec:
                reraise(suffix='(for {})'.format(self.spec.id))
            else:
                raise
                
    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        # _closed will be missing if this instance is still
        # initializing.
        if not hasattr(self, '_closed') or self._closed:
            return

        if self._owns_render:
            self.render(close=True)

        self._close()
        env_closer.unregister(self._env_closer_id)
        # If an error occurs before this line, it's possible to
        # end up with double close.
        self._closed = True                
