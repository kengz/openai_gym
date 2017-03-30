import numpy as np
from rl.policy.base_policy import Policy
from rl.util import log_self
from keras import backend as K

class ActorCriticPolicy(Policy):

    '''
    The Policy to accompany an actor-critic agent
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking
        super(ActorCriticPolicy, self).__init__()
        log_self(self)

    def softmax(self, x):
        '''Subtracting large constant from each of x values to prevent overflow'''
        # max_per_row = np.amax(x, out=None, axis=1)
        # # Issues with amax and scipy
        y = np.asarray(range(x.shape[0]))
        z = np.argmax(x, axis=1)
        max_per_row = x[y, z] 
        if (max_per_row.ndim == 1):
            max_per_row = max_per_row.reshape((max_per_row.shape[0], 1))
        x = x - max_per_row
        '''Adding small constant to prevent underflow'''
        exp_x = np.exp(x) + 0.001
        exp_x_sum = exp_x.sum(axis=1)
        exp_x_sum = exp_x_sum.reshape((exp_x_sum.shape[0],1))
        return exp_x / exp_x_sum

    def compute_action_probs(self, action_vals):
        return self.softmax(action_vals)

    def pick_action(self, action_probs):
        one_hot_actions = np.zeros((1, self.agent.env_spec['action_dim']), dtype=int)
        curr_prob = 0.0
        target = np.random.random_sample()
        for i in range(self.agent.env_spec['action_dim']):
            curr_prob += action_probs[0][i]
            if curr_prob >= target:
                one_hot_actions[0][i] = 1
                break
        # Select first action if nothing selected due to floating point errors
        if np.sum(one_hot_actions[0]) == 0:
            one_hot_actions[0][0] = 1
        assert np.sum(one_hot_actions) == action_probs.shape[0]
        action = np.argmax(one_hot_actions, axis=1)[0]
        return action

    def select_action(self, state):
        state = np.reshape(state, (1, self.agent.env_spec['state_dim']))
        action_vals = self.agent.actor.model.predict(state)
        action_probs = self.compute_action_probs(action_vals)
        action = self.pick_action(action_probs)
        return action

    def update(self, sys_vars):
        '''No updates to the policy needed'''
        pass
