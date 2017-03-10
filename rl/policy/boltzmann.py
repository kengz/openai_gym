import numpy as np
from rl.policy.base_policy import Policy
from rl.util import log_self


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
        self.clip_val = 100000
        log_self(self)

    def select_action(self, state):
        agent = self.agent
        state = np.reshape(state, (1, state.shape[0]))
        Q_state = agent.model.predict(state)[0]  # extract from batch predict
        assert Q_state.ndim == 1
        Q_state = Q_state.astype('float32')  # fix precision nan issue
        Q_state = Q_state - np.amax(Q_state)  # prevent overflow
        exp_values = np.clip(
            np.exp(Q_state / self.tau), -self.clip_val, self.clip_val)
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


class DoubleDQNBoltzmannPolicy(BoltzmannPolicy):

    '''
    Same as the Boltzmann policy but for a Double DQN agent
    '''

    def __init__(self,
                 init_tau=5., final_tau=0.5, exploration_anneal_episodes=20,
                 **kwargs):  # absorb generic param without breaking
        super(DoubleDQNBoltzmannPolicy, self).__init__()

    def select_action(self, state):
        agent = self.agent
        state = np.reshape(state, (1, state.shape[0]))
        Q_state1 = agent.model.predict(state)[0]  # extract from batch predict
        Q_state2 = agent.model2.predict(state)[0]  # extract from batch predict
        Q_state = Q_state1 + Q_state2
        assert Q_state.ndim == 1
        Q_state = Q_state.astype('float32')  # fix precision nan issue
        Q_state = Q_state - np.amax(Q_state)  # prevent overflow
        exp_values = np.clip(
            np.exp(Q_state / self.tau), -self.clip_val, self.clip_val)
        assert not np.isnan(exp_values).any()
        probs = np.array(exp_values / np.sum(exp_values))
        probs /= probs.sum()  # renormalize to prevent floating pt error
        action = np.random.choice(agent.env_spec['actions'], p=probs)
        return action
