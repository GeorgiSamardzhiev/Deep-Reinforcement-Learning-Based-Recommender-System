# -*- coding: utf-8 -*-
"""full_notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_tC9DPtNqgX4jAkLUfbV7a9pOf09Vj_L
"""

from surprise import Dataset
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random

class Env:
  def __init__(self, users_dict, users_history_lens, state_size, fix_user_id=None):

    self.users_dict = users_dict
    self.users_history_lens = users_history_lens
    self.state_size = state_size

    self.fix_user_id = fix_user_id

    self.available_users = self._generate_available_users()

    self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
    self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
    self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
    self.done = False

    self.recommended_items = set([])
    self.done_count = 10

  def _generate_available_users(self):
    available_users = []

    for i, length in zip(self.users_dict.keys(), self.users_history_lens):
      if length > self.state_size:
        available_users.append(i)

    return available_users

  def reset(self):
    self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
    self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
    self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
    self.done = False
    self.recommended_items = set([])
    return self.user, self.items, [data[0] for data in self.users_dict[self.user]], self.done

  def get_reward(self, action):
    reward = 0

    if action in self.user_items.keys():
        reward = (self.user_items[action] - 3)/2
        if action in self.recommended_items:
          reward = reward/2
        else:
          self.items = self.items[1:] + [action]
    self.recommended_items.add(action)
    return reward

  def step(self, action, step, top_k=False):
    if top_k:
      reward = list(map(lambda a: self.get_reward(a), action))
    else:
      reward = self.get_reward(action)

    if step >= self.done_count-1 or len(self.recommended_items) >= self.users_history_lens[self.user-1]: #-1??
      self.done = True

    return step+1, self.items, reward, self.done, self.recommended_items

class UserMovieEmbedding(nn.Module):
    def __init__(self, n_users, n_movies, n_factors = 100, nh = 20, p1 = 0.05, p2= 0.5):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.u.weight.data.uniform_(-0.01,0.01)
        self.m = nn.Embedding(n_movies, n_factors)
        self.m.weight.data.uniform_(-0.01,0.01)
        self.lin1 = nn.Linear(n_factors*2, nh)  # bias is True by default
        self.lin2 = nn.Linear(nh, 1)
        self.drop1 = nn.Dropout(p = p1)
        self.drop2 = nn.Dropout(p = p2)
    
    def forward(self, users, movies): # forward pass i.e.  dot product of vector from movie embedding matrixx
                                    # and vector from user embeddings matrix
        
        # torch.cat : concatenates both embedding matrix to make more columns, same rows i.e. n_factors*2, n : rows
        # u(users) is doing lookup for indexed mentioned in users
        # users has indexes to lookup in embedding matrix. 
        
        u2,m2 = self.u(users) , self.m(movies)
       
        x = self.drop1(torch.cat([u2,m2], 1)) # drop initialized weights
        x = self.drop2(F.relu(self.lin1(x))) # drop 1st linear + nonlinear wt
        r = torch.sigmoid(self.lin2(x)) * (max_rating - min_rating) + min_rating               
        return r
    
    
    def get_user_embedding_layer(self):
        return self.u
    
    def get_movie_embedding_layer(self):
        return self.m

class InnerProductLayer(nn.Module):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.
      Input shape
        - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape:
        ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.
      Arguments
        - **reduce_sum**: bool. Whether return inner product or element-wise product
      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//
            Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.]
            (https://arxiv.org/pdf/1611.00144.pdf)"""

    def __init__(self, num_inputs, device='cpu'):
        super(InnerProductLayer, self).__init__()
        self.W = nn.Parameter(torch.diag(torch.rand((num_inputs,1))))
        self.W.requires_grad = True
        self.to(device)

    def forward(self, inputs, user):

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        embed_list = torch.matmul(self.W, embed_list)
        embed_list = embed_list.unsqueeze(1)

        # create all pairs of item embeddings
        #and after that multiply element-wise
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)

        p = torch.cat([embed_list[idx]
                       for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx]
                       for idx in col], dim=1)

        #multiply element-wise the user embedding with all items
        u = user * embed_list

        inner_product = p * q

        u = u.reshape(-1).unsqueeze(0)
        result = torch.cat((u, inner_product), dim=1)
        return result

import numpy as np
from collections import deque

class SumTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.tree = np.zeros((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def update_tree(self, index):
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            self.tree[index] = self.tree[left] + self.tree[right]
            if index == 0:
                break

    def add_data(self, priority):
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.update_tree(self.index)
        self.index += 1

    def search(self, num):
        current = 0
        while True:
            left = (current * 2) + 1
            right = (current * 2) + 2

            if num <= self.tree[left]:
                current = left
            else:
                num -= self.tree[left]
                current = right
            
            if current >= self.buffer_size - 1:
                break

        return self.tree[current], current, current - self.buffer_size + 1

    def update_prioirty(self, priority, index):
        self.tree[index] = priority
        self.update_tree(index)

    def sum_all_prioirty(self):
        return float(self.tree[0])


class MinTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.tree = np.ones((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def update_tree(self, index):
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            if self.tree[left] > self.tree[right]:
                self.tree[index] = self.tree[right]
            else:
                self.tree[index] = self.tree[left]
            if index == 0:
                break

    def add_data(self, priority):
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.update_tree(self.index)
        self.index += 1

    def update_prioirty(self, priority, index):
        self.tree[index] = priority
        self.update_tree(index)

    def min_prioirty(self):
        return float(self.tree[0])

class PriorityExperienceReplay(object):

    '''
    apply PER
    '''

    def __init__(self, buffer_size, embedding_dim, state_size):
        self.buffer_size = buffer_size
        self.crt_idx = 0
        self.is_full = False
        
        '''
            state : (300,), 
            next_state : (300,) 변할 수 잇음, 
            actions : (100,), 
            rewards : (1,), 
            dones : (1,)
        '''
        self.states = torch.zeros((buffer_size, state_size))
        self.actions = torch.zeros((buffer_size, embedding_dim))
        self.rewards = torch.zeros((buffer_size))
        self.next_states = torch.zeros((buffer_size, state_size))
        self.dones = torch.zeros(buffer_size)

        self.sum_tree = SumTree(buffer_size)
        self.min_tree = MinTree(buffer_size)

        self.max_prioirty = 1.0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_constant = 0.00001

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.sum_tree.add_data(self.max_prioirty ** self.alpha)
        self.min_tree.add_data(self.max_prioirty ** self.alpha)
        
        self.crt_idx = (self.crt_idx + 1) % self.buffer_size
        if self.crt_idx == 0:
            self.is_full = True

    def sample(self, batch_size):
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_prioirty()
        
        N = self.buffer_size if self.is_full else self.crt_idx
        min_priority = self.min_tree.min_prioirty() / sum_priority
        max_weight = (N * min_priority) ** (-self.beta)

        segment_size = sum_priority/batch_size
        for j in range(batch_size):
            min_seg = segment_size * j
            max_seg = segment_size * (j + 1)

            random_num = random.uniform(min_seg, max_seg)
            priority, tree_index, buffer_index = self.sum_tree.search(random_num)
            rd_idx.append(buffer_index)

            p_j = priority / sum_priority
            w_j = (p_j * N) ** (-self.beta) / max_weight
            weight_batch.append(w_j)
            index_batch.append(tree_index)
        self.beta = min(1.0, self.beta + self.beta_constant)

        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, torch.Tensor(weight_batch), index_batch

    def update_priority(self, priority, index):
        self.sum_tree.update_prioirty(priority ** self.alpha, index)
        self.min_tree.update_prioirty(priority ** self.alpha, index)
        self.update_max_priority(priority ** self.alpha)

    def update_max_priority(self, priority):
        self.max_prioirty = max(self.max_prioirty, priority)

class CriticNetwork(nn.Module):
  def __init__(self, state_size, action_size):
    super(CriticNetwork, self).__init__()

    self.lin1 = nn.Linear(state_size, state_size)
    self.lin2 = nn.Linear(state_size+action_size, state_size)
    self.lin3 = nn.Linear(state_size, 1)

    self.relu = nn.ReLU()

  def forward(self, action, state):
    state = self.relu(self.lin1(state))
    input_concat = torch.cat((action, state), dim=1)

    x = self.lin2(input_concat)
    x = self.relu(x)
    x = self.lin3(x)
    x = self.relu(x)

    return x

class Critic:
  def __init__(self, embedding_dim, hidden_dim, state_size, learning_rate, tau):
    self.embedding_dim = embedding_dim

    self.local_network = CriticNetwork(state_size, embedding_dim)
    self.target_network = CriticNetwork(state_size, embedding_dim)

    self.optimizer = torch.optim.Adam(self.local_network.parameters(),lr=learning_rate)
    self.loss = nn.MSELoss()

    self.tau = tau

  def update_target_network(self):
    """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
    """

    for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
      target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


  def dq_da(self, input):
    """
      Gradient of Q at a
    """


  def train(self, actions,states, td_targets, weight_batch):
    with torch.autograd.set_detect_anomaly(True):
      self.optimizer.zero_grad()
      outputs = self.local_network(actions, states)
      loss = self.loss(outputs.detach(), td_targets)
      loss = torch.mean(weight_batch*loss)

      loss.backward(retain_graph=True)
      self.optimizer.step()
      return loss

class ActorNetwork(nn.Module):
  def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()        
        self.lin1 = nn.Linear(in_features=state_dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.lin3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

  def forward(self, state):
        new_state = torch.relu(self.lin1(state))
        new_state = torch.relu(self.lin2(new_state))
        action = torch.tanh(self.lin3(new_state))
        return action

class Actor:
  def __init__(self, embedding_dim, hidden_dim, state_size, learning_rate, tau):
    self.embedding_dim = embedding_dim
    self.state_size = state_size

    self.local_network = ActorNetwork(state_size, hidden_dim, embedding_dim)
    self.target_network = ActorNetwork(state_size, hidden_dim, embedding_dim)

    self.optimizer = torch.optim.Adam(self.local_network.parameters(),lr=learning_rate)
    self.loss = nn.MSELoss()

    self.tau = tau

  def train(self):
    pass


  def update_target_network(self):
    """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
    """

    for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
      target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

class DrrAveState(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=1, kernel_size=1)
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=1)
        
    def forward(self, item_embeddings, user_embedding):
        drr_ave = self.conv(item_embeddings)
        drr_ave = self.avg_pool(drr_ave).squeeze(1).T
        return torch.cat((user_embedding, torch.mul(user_embedding, drr_ave), drr_ave), 1)

class Recommender:

  EMBEDDING_DIM = 100

  ACTOR_HIDDEN_DIM = 128
  ACTOR_LR = 0.001
  CRITIC_HIDDEN_DIM = 128
  CRITIC_LR = 0.001

  DISCOUNT_FACTOR = 0.9
  TAU = 0.001

  REPLAY_MEMORY_SIZE = 1000
  BATCH_SIZE = 32

  EPSILON_FOR_PRIORITY = 1e-6

  def __init__(self, env, users, items, state_size, ):
    self.env = env
    self.users = users
    self.items = items

    self.state_size = int(3*self.EMBEDDING_DIM) #drr-ave output dim

    self.actor = Actor(self.EMBEDDING_DIM, self.ACTOR_HIDDEN_DIM, self.state_size, self.ACTOR_LR, self.TAU)
    self.critic = Critic(self.EMBEDDING_DIM, self.CRITIC_HIDDEN_DIM, self.state_size, self.CRITIC_LR, self.TAU)
    # self.actor.local_network.load_state_dict(torch.load('actor_local'))
    # self.actor.target_network.load_state_dict(torch.load('actor_target'))
    # self.critic.local_network.load_state_dict(torch.load('critic_local'))
    # self.critic.target_network.load_state_dict(torch.load('critic_target'))

    self.embedding_network = UserMovieEmbedding(max(self.users)+1, max(self.items)+1, self.EMBEDDING_DIM)
    self.embedding_network.load_state_dict(torch.load('saved_concat_model'))

    #state representation of user and item embeddings
    self.state_repr = DrrAveState(self.EMBEDDING_DIM)

    #initialize PER buffer
    self.buffer = PriorityExperienceReplay(self.REPLAY_MEMORY_SIZE, self.EMBEDDING_DIM, self.state_size)
    
    self.epsilon = 1
    self.epsilon_decay = 0.9999
    self.std = 1.5

  def recommend_item(self, action, item_ebs, items, top_k=False):
    action = torch.transpose(action, 0, 1)

    if top_k:
      product = torch.mm(item_ebs, action)
      item_idx = torch.argsort(product, axis=0,descending=True)[-top_k:]
      return torch.index_select(torch.Tensor(items), 0, item_idx.flatten()).numpy()
    else:
      product = torch.mm(item_ebs, action)
      item_idx = torch.argmax(product, axis=0)
      return items[item_idx]

  def calculate_td_targets(self, rewards, q_values, dones):
    y_t = torch.clone(q_values)
    for i in range(q_values.shape[0]):
      y_t[i] = rewards[i] + (1-dones[i])*self.DISCOUNT_FACTOR*q_values[i]

    return y_t

  def train(self, max_episode_num, top_k=False):

    self.actor.update_target_network()
    self.critic.update_target_network()

    for episode in range(max_episode_num):
      #init variables
      value_loss = 0
      episode_reward = 0
      correct_count = 0
      mean_action = 0

      user_id, items_ids, all_items_ids, done = self.env.reset()
      step = 0
      rewards = []
      policy_losses = []
      all_items_embeddings = self.embedding_network.m(torch.LongTensor(list(all_items_ids)))
      while not done:
        #get user embedding
        user_embedding = self.embedding_network.u(torch.LongTensor([user_id]))

        #get item embeddings
        item_embeddings = self.embedding_network.m(torch.LongTensor(list(items_ids)))

        #get state representation
        state = self.state_repr(item_embeddings.unsqueeze(-1), user_embedding)

        action = self.actor.local_network(state)

        #epsilon-greedy exploration
        if self.epsilon > np.random.uniform():
          #epsilon decay?
          self.epsilon *= self.epsilon_decay
          action += torch.randn(size=action.shape)

        #Recommended item
        recommended_item = self.recommend_item(action, all_items_embeddings, all_items_ids, top_k=top_k)

        #observe new state and get reward
        step, next_item_ids, reward, done, recommended_items = self.env.step(recommended_item, step, top_k=top_k)
        if top_k:
          reward = np.sum(reward)

        rewards.append(reward)

        #get next item embedding
        next_items_embeddings = self.embedding_network.m(torch.LongTensor(next_item_ids))

        #get next state representation
        next_state = self.state_repr(next_items_embeddings.unsqueeze(-1), user_embedding) #add one additional dimension because of the input shape

        #add in buffer current transition
        self.buffer.append(state, action, reward, next_state, done)

        if self.buffer.crt_idx > 1 or self.buffer.is_full:

          #sample minibatch
          batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, \
            weight_batch, index_batch = self.buffer.sample(self.BATCH_SIZE)

          batch_target_next_actions = self.actor.target_network(batch_next_states)
          next_q_values = self.critic.target_network(batch_target_next_actions, batch_next_states)

          #calculate td targets
          td_targets = self.calculate_td_targets(batch_rewards, next_q_values, batch_dones)

          #update priority
          for (p, i) in zip(td_targets, index_batch):
            self.buffer.update_priority(abs(p[0]) + self.EPSILON_FOR_PRIORITY, i)

          # batch_actions?
          value_loss += self.critic.train(batch_actions, batch_states, td_targets, weight_batch)

          self.actor.optimizer.zero_grad()
          actions = self.actor.local_network(batch_states)
          policy_loss = -self.critic.local_network(actions.detach(), batch_states)
          policy_loss = policy_loss.mean()

          policy_loss.backward(retain_graph=True)
          policy_losses.append(policy_loss)
          self.actor.optimizer.step()

          #Soft update
          self.actor.update_target_network()
          self.critic.update_target_network()

        episode_reward += reward

        mean_action += torch.sum(action[0])/(len(action[0]))

        if reward > 0:
          correct_count += 1
      print("EPISODE_END avg:", np.sum(rewards)/step)
      print("EPISODE_END abg_polisy_loss:", np.sum(policy_losses)/step)
      torch.save(self.actor.local_network.state_dict(), "actor_local")
      torch.save(self.actor.target_network.state_dict(), "actor_target")
      torch.save(self.critic.local_network.state_dict(), "critic_local")
      torch.save(self.critic.target_network.state_dict(), "critic_target")
      print('value_loss', value_loss/step)

data = Dataset.load_builtin('ml-1m')
#trainset = data.build_full_trainset()
#users_num = trainset.n_users
#items_num = trainset.n_items

df = pd.DataFrame(data.raw_ratings, columns = ['UserId', 'MovieId', 'Rating',  'Timestamp'], dtype='int32')
df = df.astype('int32')
users = df['UserId'].unique()
items = df['MovieId'].unique()

#Arranged in order of the movies watched by users
users_dict = np.load('user_dict.npy', allow_pickle=True)


#Movie history length for each user
users_history_lens = np.load('users_histroy_len.npy')

# Training setting
train_users_num = int(len(users) * 0.8)
train_items_num = len(items) 
train_users_dict = {k:users_dict.item().get(k) for k in range(1, train_users_num+1)}
train_users_history_lens = users_history_lens[:train_users_num]

STATE_SIZE = 100
MAX_EPISODE_NUM = 8000

env = Env(train_users_dict, train_users_history_lens, STATE_SIZE)

print('before algo')
recommender = Recommender (env, users, items, STATE_SIZE)

print('before train')
recommender.train(MAX_EPISODE_NUM)