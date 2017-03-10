from rl.agent.dqn import DQN


class DeepSarsa(DQN):

    '''
    Deep Sarsa agent.
    On policy, with updates after each experience
    Policy = epsilonGreedyPolicy
    '''

    def __init__(self, *args, **kwargs):
        super(DeepSarsa, self).__init__(*args, **kwargs)
        self.train_per_n_new_exp = 1
        self.batch_size = 1
        self.n_epoch = 1
        self.final_n_epoch = 1

    def compute_Q_states(self, last_exp):
        (Q_states, Q_next_states, _max) = super(
            DeepSarsa, self).compute_Q_states(last_exp)
        next_action = self.select_action(last_exp['next_states'][0])
        Q_next_states_selected = Q_next_states[:, next_action]
        return (Q_states, Q_next_states, Q_next_states_selected)

    def train_an_epoch(self):
        last_exp = self.memory.pop()
        (Q_states, _next, Q_next_states_selected
         ) = self.compute_Q_states(last_exp)
        Q_targets = self.compute_Q_targets(
            last_exp, Q_states, Q_next_states_selected)
        loss = self.model.train_on_batch(last_exp['states'], Q_targets)
        return loss
