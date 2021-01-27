#This code is written by @sracaniere from DeepMind
#https://github.com/sracaniere

import numpy as np
import math

class TwoArmedBandit(object):
  def __init__(self, frame_cap=300):
    self.frame_cap = frame_cap
    self.state = 0
    self.frame = 0
    self.reward = 0.
    self.pcontinue = 1
    self.timer = 0
    self.step_reward = 0

  def start(self):
    """Starts a new episode."""
    self.state = 0
    self.frame = 0
    self.reward = 0
    self.pcontinue = 1
    return self._state2onehot(self.state), self.reward, self.pcontinue

  def step(self, action):
    """Advances environment one time-step following the given action."""
    self.frame += 1
    self.pcontinue = 1
    self.reward = self.step_reward
    self.timer += 1
    
    if(self.state == 0):
        if(action == 0):
            self.state = 1
        else:
            self.state = 2
    else:  
        if(self.state == 1):
            self.reward += 1
        else:
            self.reward += 0
        self.state = 0        

    # Check if framecap reached
    if self.frame_cap > 0 and self.frame >= self.frame_cap:
      self.pcontinue = 0

  def observation(self, agent_id=0):
    return (self.reward,
            self.pcontinue,
            self._state2onehot(self.state))

  def _state2onehot(self, x):
        x_onehot = np.zeros(3)
        x_onehot[x] = 1
        return x_onehot
    
class TwoArmedBanditUnc08(object):
  def __init__(self, frame_cap=300):
    self.frame_cap = frame_cap
    self.state = 0
    self.frame = 0
    self.reward = 0.
    self.pcontinue = 1
    self.timer = 0
    self.step_reward = 0

  def start(self):
    """Starts a new episode."""
    self.state = 0
    self.frame = 0
    self.reward = 0
    self.pcontinue = 1
    return self._state2onehot(self.state), self.reward, self.pcontinue

  def step(self, action):
    """Advances environment one time-step following the given action."""
    self.frame += 1
    self.pcontinue = 1
    self.reward = self.step_reward
    self.timer += 1
    
    if(self.state == 0):
        prob = np.random.uniform()
        if(action == 0):
            if prob <= 0.8:
                self.state = 1
            else:
                self.state = 2
        else:
            if prob <= 0.8:
                self.state = 2
            else:
                self.state = 1
    else:  
        if(self.state == 1):
            self.reward += 1
        else:
            self.reward += 0
        self.state = 0        

    # Check if framecap reached
    if self.frame_cap > 0 and self.frame >= self.frame_cap:
      self.pcontinue = 0

  def observation(self, agent_id=0):
    return (self.reward,
            self.pcontinue,
            self._state2onehot(self.state))

  def _state2onehot(self, x):
        x_onehot = np.zeros(3)
        x_onehot[x] = 1
        return x_onehot

class TwoArmedBanditUnc06(object):
  def __init__(self, frame_cap=300):
    self.frame_cap = frame_cap
    self.state = 0
    self.frame = 0
    self.reward = 0.
    self.pcontinue = 1
    self.timer = 0
    self.step_reward = 0

  def start(self):
    """Starts a new episode."""
    self.state = 0
    self.frame = 0
    self.reward = 0
    self.pcontinue = 1
    return self._state2onehot(self.state), self.reward, self.pcontinue

  def step(self, action):
    """Advances environment one time-step following the given action."""
    self.frame += 1
    self.pcontinue = 1
    self.reward = self.step_reward
    self.timer += 1
    
    if(self.state == 0):
        prob = np.random.uniform()
        if(action == 0):
            if prob <= 0.6:
                self.state = 1
            else:
                self.state = 2
        else:
            if prob <= 0.6:
                self.state = 2
            else:
                self.state = 1
    else:  
        if(self.state == 1):
            self.reward += 1
        else:
            self.reward += 0
        self.state = 0        

    # Check if framecap reached
    if self.frame_cap > 0 and self.frame >= self.frame_cap:
      self.pcontinue = 0

  def observation(self, agent_id=0):
    return (self.reward,
            self.pcontinue,
            self._state2onehot(self.state))

  def _state2onehot(self, x):
        x_onehot = np.zeros(3)
        x_onehot[x] = 1
        return x_onehot