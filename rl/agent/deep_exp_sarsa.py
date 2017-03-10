import numpy as np
from rl.agent.deep_sarsa import DeepSarsa


class DeepExpectedSarsa(DeepSarsa):

    '''
    Deep Expected Sarsa agent.
    On policy, with updates after each experience
    Policy = epsilonGreedyPolicy
    '''

    def compute_Q_states(self, last_exp):
        (Q_states, Q_next_states, _max) = super(
            DeepExpectedSarsa, self).compute_Q_states(last_exp)

        curr_e = self.policy.e
        curr_e_per_a = curr_e / self.env_spec['action_dim']

        Q_next_states_max = np.amax(Q_next_states, axis=1)
        Q_next_states_selected = (1 - curr_e) * Q_next_states_max + \
            np.sum(Q_next_states * curr_e_per_a, axis=1)
        return (Q_states, Q_next_states, Q_next_states_selected)
