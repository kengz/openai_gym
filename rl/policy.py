import numpy as np
from rl.util import log_self
from keras import backend as K

class Policy(object):

    '''
    The base class of Policy, with the core methods
    '''

    def __init__(self,
                 **kwargs):  # absorb generic param without breaking
        '''Construct externally, and set at Agent.compile()'''
        self.agent = None

    def select_action(self, state):
        raise NotImplementedError()

    def update(self, sys_vars):
        raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):

    '''
    The Epsilon-greedy policy
    '''

    def __init__(self,
                 init_e=1.0, final_e=0.1, exploration_anneal_episodes=30,
                 **kwargs):  # absorb generic param without breaking
        super(EpsilonGreedyPolicy, self).__init__()
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.exploration_anneal_episodes = exploration_anneal_episodes
        log_self(self)

    def select_action(self, state):
        '''epsilon-greedy method'''
        agent = self.agent
        if self.e > np.random.rand():
            action = np.random.choice(agent.env_spec['actions'])
        else:
            state = np.reshape(state, (1, state.shape[0]))
            # extract from batch predict
            Q_state = agent.model.predict(state)[0]
            assert Q_state.ndim == 1
            action = np.argmax(Q_state)
        return action

    def update(self, sys_vars):
        '''strategy to update epsilon in agent'''
        epi = sys_vars['epi']
        rise = self.final_e - self.init_e
        slope = rise / float(self.exploration_anneal_episodes)
        self.e = max(slope * epi + self.init_e, self.final_e)
        return self.e


class OscillatingEpsilonGreedyPolicy(EpsilonGreedyPolicy):

    '''
    The epsilon-greedy policy with oscillating epsilon
    periodically agent.e will drop to a fraction of
    the current exploration rate
    '''

    def update(self, sys_vars):
        '''strategy to update epsilon in agent'''
        super(OscillatingEpsilonGreedyPolicy, self).update(
            sys_vars)
        epi = sys_vars['epi']
        if not (epi % 3) and epi > 15:
            # drop to 1/3 of the current exploration rate
            self.e = max(self.e/3., self.final_e)
        return self.e


class TargetedEpsilonGreedyPolicy(EpsilonGreedyPolicy):

    '''
    switch between active and inactive exploration cycles by
    partial mean rewards and its distance to the target mean rewards
    '''

    def update(self, sys_vars):
        '''strategy to update epsilon in agent'''
        epi = sys_vars['epi']
        SOLVED_MEAN_REWARD = sys_vars['SOLVED_MEAN_REWARD']
        REWARD_MEAN_LEN = sys_vars['REWARD_MEAN_LEN']
        PARTIAL_MEAN_LEN = int(REWARD_MEAN_LEN * 0.20)
        if epi < 1:  # corner case when no total_rewards_history to avg
            return
        # the partial mean for projection the entire mean
        partial_mean_reward = np.mean(
            sys_vars['total_rewards_history'][-PARTIAL_MEAN_LEN:])
        # difference to target, and its ratio (1 if denominator is 0)
        min_reward = np.amin(sys_vars['total_rewards_history'])
        projection_gap = SOLVED_MEAN_REWARD - partial_mean_reward
        worst_gap = SOLVED_MEAN_REWARD - min_reward
        gap_ratio = projection_gap / worst_gap
        envelope = self.init_e + (self.final_e - self.init_e) / 2. * \
            (float(epi)/float(self.exploration_anneal_episodes))
        pessimistic_gap_ratio = envelope * min(2 * gap_ratio, 1)
        # if is in odd cycle, and diff is still big, actively explore
        active_exploration_cycle = not bool(
            int(epi/PARTIAL_MEAN_LEN) % 2) and (
            projection_gap > abs(SOLVED_MEAN_REWARD * 0.05))
        self.e = max(pessimistic_gap_ratio * self.init_e, self.final_e)

        if not active_exploration_cycle:
            self.e = max(self.e/2., self.final_e)
        return self.e


class BoltzmannPolicy(Policy):

    '''
    The Boltzmann policy, where prob dist for selection
    p = exp(Q/tau) / sum(Q[a]/tau)
    '''

    def __init__(self,
                 init_tau=5., final_tau=0.5, exploration_anneal_episodes=20,
                 **kwargs):  # absorb generic param without breaking
        super(BoltzmannPolicy, self).__init__()
        self.init_tau = init_tau
        self.final_tau = final_tau
        self.tau = self.init_tau
        self.exploration_anneal_episodes = exploration_anneal_episodes
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.reshape(state, (1, state.shape[0]))
        Q_state = agent.model.predict(state)[0]  # extract from batch predict
        assert Q_state.ndim == 1
        Q_state = Q_state.astype('float64')  # fix precision nan issue
        Q_state = Q_state - np.amax(Q_state)  # prevent overflow
        exp_values = np.exp(Q_state / self.tau) + 0.0001  # prevent underflow
        assert not np.isnan(exp_values).any()
        probs = np.array(exp_values / np.sum(exp_values))
        probs /= probs.sum()  # renormalize to prevent floating pt error
        action = np.random.choice(agent.env_spec['actions'], p=probs)
        return action

    def update(self, sys_vars):
        '''strategy to update tau in agent'''
        epi = sys_vars['epi']
        rise = self.final_tau - self.init_tau
        slope = rise / float(self.exploration_anneal_episodes)
        self.tau = max(slope * epi + self.init_tau, self.final_tau)
        return self.tau

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
