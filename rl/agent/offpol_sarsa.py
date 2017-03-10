import numpy as np
from rl.agent.dqn import DQN


class OffPolicySarsa(DQN):

    '''
    Deep Sarsa agent.
    Off policy. Reduces to Q learning when eval_e = 0
    Evaluation policy = epsilonGreedyPolicy, eval_e = 0.05
    Experience generating policy = Boltzmann or
    EpsilonGreedy with annealing
    '''

    def __init__(self, *args, **kwargs):
        super(OffPolicySarsa, self).__init__(*args, **kwargs)
        self.eval_e = 0.05

    def compute_Q_states(self, last_exp):
        (Q_states, Q_next_states, _max) = super(
            OffPolicySarsa, self).compute_Q_states(last_exp)

        e_per_action = self.eval_e / self.env_spec['action_dim']

        Q_next_states_max = np.amax(Q_next_states, axis=1)
        expected_Q = (1 - self.eval_e) * Q_next_states_max + \
            np.sum(Q_next_states * e_per_action, axis=1)
        return (Q_states, None, expected_Q)
