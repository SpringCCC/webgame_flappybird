import numpy as np
from configs import config


class ReplayBuffer():
    
    def __init__(self, state_dim=2, max_buffer=config.max_buffer, batch_size=config.batch_size):
        self.state = np.zeros([max_buffer, state_dim], dtype=np.int32)
        self.next_state = np.zeros([max_buffer, state_dim], dtype=np.int32)
        self.reward = np.zeros([max_buffer], dtype=np.float32)
        self.action = np.zeros([max_buffer], dtype=np.int32)
        self.done = np.zeros([max_buffer], dtype=np.float32)
        self.max_size = max_buffer
        self.cur_ptr = 0
        self.cur_size = 0
        self.batch_size = batch_size
        
        
    def store(self, s, a, r, s1, done):
        self.state[self.cur_ptr] = np.asarray(s)
        self.next_state[self.cur_ptr] = np.asarray(s1)
        self.reward[self.cur_ptr] = r
        self.action[self.cur_ptr] = a
        self.done[self.cur_ptr] = done
        self.cur_size = min(self.max_size, self.cur_size+1)
        self.cur_ptr = (self.cur_ptr+1)%self.max_size

        
    def sample(self, bs=64):
        idx = np.random.choice(self.cur_size, self.batch_size, replace=False)
        trans = {}
        trans['state'] = self.state[idx]
        trans['next_state'] = self.next_state[idx]
        trans['reward'] = self.reward[idx]
        trans['action'] = self.action[idx]
        trans['done'] = self.done[idx]
        return trans
        
    def size(self):
        return self.cur_size