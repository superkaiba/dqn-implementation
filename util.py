import torch
import numpy as np

import queue

class History:
    def __init__(self, *, stacklen: int, device):
        self._device = device
        self._stacklen = stacklen
        self.reset()

    def get_history(self):
      return (
        torch.tensor(np.array(list(self._scent.queue)), device=self._device),
        torch.tensor(np.array(list(self._vision.queue)), device=self._device),
        torch.tensor(np.array(list(self._feature.queue)), device=self._device),
        torch.tensor(np.array(list(self._action.queue)), device=self._device),
      )
    def add_obs(self, obs):
      self._scent.get()
      self._vision.get()
      self._feature.get()

      self._scent.put(obs[0]),
      self._vision.put(obs[1])
      self._feature.put(obs[2].reshape((15,15,4)).astype(np.float32))

    def add_action(self, action):
      self._action.get()
      self._action.put(action)

    def reset(self):
        self._scent = queue.Queue(maxsize=self._stacklen)
      
        for i in range(self._stacklen):
          self._scent.put(np.zeros((3,), dtype=np.float32))
        
        self._vision = queue.Queue(maxsize=self._stacklen)
        for i in range(self._stacklen):
          self._vision.put(np.zeros((15, 15, 3), dtype=np.float32))
        
        self._feature = queue.Queue(maxsize=self._stacklen)
        for i in range(self._stacklen):
          self._feature.put(np.zeros((15, 15,4), dtype=np.float32))

        self._action = queue.Queue(maxsize=self._stacklen)
        for i in range(self._stacklen):
          self._action.put(np.zeros((4,), dtype=np.float32))
        self._reward = queue.Queue(maxsize=self._stacklen)

    def __len__(self):
        return self._scent.qsize()
