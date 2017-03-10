import numpy as np
from rl.agent.base_agent import Agent


class Dummy(Agent):

    '''
    A dummy agent that does random actions, for demo
    '''

    def select_action(self, state):
        '''epsilon-greedy method'''
        action = np.random.choice(self.env_spec['actions'])
        return action

    def update(self, sys_vars):
        return

    def to_train(self, sys_vars):
        return True

    def train(self, sys_vars):
        return

    def compile_model(self):
        return


class QTable(Agent):

    '''
    The simplest Q learner - a table,
    with epsilon-greedy method and
    Bellman equation for value.
    '''

    def __init__(self, env_spec,
                 resolution=10,
                 gamma=0.95, lr=0.1,
                 init_e=1.0, final_e=0.1, exploration_anneal_episodes=1000,
                 **kwargs):  # absorb generic param without breaking
        super(QTable, self).__init__(env_spec)
        self.resolution = resolution
        self.gamma = gamma
        self.lr = lr
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.exploration_anneal_episodes = exploration_anneal_episodes
        self.build_table()

    def build_table(self):
        '''
        init the 2D qtable by
        bijecting the state space into pixelated, flattened vector
        multiplied with
        list of possible discrete actions
        '''
        self.pixelate_state_space(self.resolution)
        flat_state_size = self.resolution ** self.env_spec['state_dim']
        self.qtable = np.random.uniform(
            low=-1, high=1,
            size=(flat_state_size, self.env_spec['action_dim']))
        return self.qtable

    def compile_model(self):
        return

    def pixelate_state_space(self, resolution=10):
        '''chunk up the state space hypercube to specified resolution'''
        state_bounds = self.env_spec['state_bounds']
        self.state_pixels = [np.linspace(*sb, num=resolution+1)
                             for sb in state_bounds]
        return self.state_pixels

    def flatten_state(self, state):
        '''
        collapse a hyperdim state by binning into state_pixels
        then flattening the pixel_state into 1-dim bijection
        '''
        val_space_pairs = list(zip(state, self.state_pixels))
        pixel_state = [np.digitize(*val_space)
                       for val_space in val_space_pairs]  # binning
        flat_state = int("".join([str(ps) for ps in pixel_state]))
        return flat_state

    def select_action(self, state):
        '''epsilon-greedy method'''
        if self.e > np.random.rand():
            action = np.random.choice(self.env_spec['actions'])
        else:
            flat_state = self.flatten_state(state)
            action = np.argmax(self.qtable[flat_state, :])
        return action

    def update_e(self):
        '''strategy to update epsilon'''
        self.e = max(self.e -
                     (self.init_e - self.final_e) /
                     float(self.exploration_anneal_episodes),
                     self.final_e)
        return self.e

    def update(self, sys_vars):
        self.update_e()

    def to_train(self, sys_vars):
        return True

    def train(self, sys_vars):
        '''
        run the basic bellman equation update
        '''
        last_exp = self.memory.pop()
        state = last_exp['states'][0]
        flat_state = self.flatten_state(state)
        next_state = last_exp['next_states'][0]
        next_flat_state = self.flatten_state(next_state)
        action = np.argmax(last_exp['actions'][0])  # from one-hot
        reward = last_exp['rewards'][0]
        Q_state_action = self.qtable[flat_state, action]
        Q_next_state = self.qtable[next_flat_state, :]
        Q_next_state_max = np.amax(Q_next_state)
        loss = (reward + self.gamma * Q_next_state_max - Q_state_action)
        sys_vars['loss'].append(loss)

        self.qtable[flat_state, action] = Q_state_action + \
            self.lr * loss
        return self.qtable
