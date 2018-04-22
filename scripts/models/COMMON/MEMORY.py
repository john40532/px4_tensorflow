import numpy as np
import copy

class SumTree:
  def __init__(self, capacity):
    assert capacity & (capacity-1) == 0
    self.tree = np.zeros(2*capacity - 1)
    self.capacity = capacity
    self._counter = 0

  def add(self, p):
    tree_idx = self._counter + self.capacity-1
    self.update(tree_idx, p)
    self._counter+=1
    if self._counter == self.capacity: self._counter = 0

  def update(self, tree_idx, p):
    change = p - self.tree[tree_idx]
    self.tree[tree_idx] = p
    while tree_idx != 0:
      tree_idx = (tree_idx -1) // 2
      self.tree[tree_idx] += change

  def get(self, v):
    parent_idx = 0
    while True:
      cl_idx = 2*parent_idx + 1
      cr_idx = cl_idx + 1
      if cl_idx >= len(self.tree):
        leaf_idx = parent_idx
        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx]
      else:
        if v <= self.tree[cl_idx]:
          parent_idx = cl_idx
        else:
          v -= self.tree[cl_idx]
          parent_idx = cr_idx

  @property
  def total_P(self):
    return self.tree[0]

  @property
  def max_P(self):
    return np.max(self.tree[-self.capacity:])

  @property
  def min_P(self):
    return np.min(self.tree[-self.capacity:])

class Memory:
  '''
  Uniform sampling batch
  '''
  def __init__(
      self, 
      state_dim,
      action_dim,
      capacity, 
      mini_batch
  ):
    self.states = np.zeros((capacity, state_dim), dtype=np.float32)
    self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
    self.rewards = np.zeros(capacity, dtype=np.float32)
    self.state_s = np.zeros((capacity, state_dim), dtype=np.float32)
    self.terms   = np.zeros(capacity, dtype=np.float32)
    
    self.capacity = capacity
    self.mini_batch = mini_batch 
    self._count = 0
    self._full = False
    
  def store(
      self,
      state, 
      action, 
      reward, 
      state_,
      term
  ):
    self.states[self._count] = state
    self.state_s[self._count] = state_
    self.actions[self._count] = action
    self.rewards[self._count] = reward
    self.terms[self._count] = term

    self._count+=1
    if self._count == self.capacity:
      self._full = True
      self._count = 0

  def random_idxs(self):
    return np.random.choice(self.capacity, size=self.mini_batch)

  def batch(self, idxs):
    assert self._full, "short memory"

    return (self.states[idxs], 
            self.actions[idxs], 
            self.rewards[idxs], 
            self.state_s[idxs],
            self.terms[idxs])

  @property
  def Full(self):
    return self._full


class PrioritizedMemory(Memory):
  '''
  Prioritized sampling batch
  '''
  def __init__(self,
    state_dim,
    action_dim,
    capacity=2**14,
    mini_batch=2**5
  ):
    Memory.__init__(self, state_dim, action_dim, capacity, mini_batch)
    self.priTree = SumTree(capacity)
    
    self._epsilon = 0.001
    #TODO parameterize beta alpha
    self._alpha = 1

  def store(self,
    state,
    action,
    reward,
    state_,
    done,
    p = None
  ):
    Memory.store(self, state, action, reward, state_, done)
    if p is None:
      if self.priTree.total_P < self._epsilon*self.capacity:
        p = 2*self._epsilon
      else:
        p = self.priTree.total_P/self.capacity
    else:
      p+=self._epsilon
    self.priTree.add(p**self._alpha)

  def prioritizedSampling(self, beta=1):
    total_P = self.priTree.total_P
    
    deltaX = total_P/self.mini_batch
    base = np.arange(self.mini_batch)
    random_v = deltaX*(np.random.uniform(self.mini_batch) + base)
    tmp = np.array([self.priTree.get(v) for v in random_v])
    idxs = tmp[:,0]
    prop = tmp[:,1]/total_P
    ISweights = np.power(prop*self.capacity, -beta)
    ISweights/=np.max(ISweights)
    return idxs, ISweights

  def updatePriority(self, idxs, Pri):
    for i,p in zip(idxs, Pri):
      p+=self._epsilon
      tree_idx = i + self.capacity-1
      self.priTree.update(tree_idx, p**self._alpha)

if __name__ == "__main__":
  obs_dim = 2
  act_dim = 1
  capacity = 2**2

  testMEM = PrioritizedMemory(obs_dim, act_dim, capacity, 4)
  i=1
  while not testMEM.Full:
    obs = (i,i)
    obs_ = (i,i)
    act = (i)
    testMEM.store(obs, act, i, obs_, False)
    i+=1

  idxs, ISW = testMEM.prioritizedSampling()
  print("idxs:\n", idxs)
  print("ISW:\n", ISW)
  print(testMEM.priTree.tree)
  print(testMEM.priTree.max_P, testMEM.priTree.min_P)
  states , actions, rewards, state_s, terms = testMEM.random_batch()
  print("states:\n", states,
        "\nactions:\n", actions,
        "\nrewards:\n", rewards,
        "\nstate_s:\n", state_s,
        "\nterms:\n", terms)
