import numpy as np
from os import getpid
from rl.agent.double_dqn import DoubleDQN
from rl.agent.dqn import DQN
from rl.util import logger


class FreezeDQN(DoubleDQN):

    '''
    Extends DQN agent to freeze target Q network
    and periodically update them to the weights of the
    exploration model
    Avoids oscillations and breaks correlation
    between Q-network and target
    http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf
    Exploration model periodically saved and loaded into target Q network
    '''

    def compute_Q_states(self, minibatch):
        Q_states = np.clip(self.model.predict(minibatch['states']),
                           -self.clip_val, self.clip_val)
        Q_next_states = np.clip(self.model2.predict(minibatch['next_states']),
                                -self.clip_val, self.clip_val)
        Q_next_states_max = np.amax(Q_next_states, axis=1)
        return (Q_states, Q_next_states, Q_next_states_max)

    def train_an_epoch(self):
        # Should call DQN to train an epoch, not DoubleDQN
        return DQN.train_an_epoch(self)

    def update_target_model(self):
        # TODO fix to not use frequent filesave, will cause memleak
        # Also, loading logic seems off
        pid = getpid()
        filename = 'temp_Q_model_freeze_' + str(pid) + '.h5'
        model_dir = 'rl/asset/model'
        modelpath = '{}/{}'.format(model_dir, filename)
        self.model.save(modelpath)
        self.model2 = self.load_model(modelpath)
        logger.debug("Updated target model weights")

    def update(self, sys_vars):
        '''
        Agent update apart from training the Q function
        '''
        done = sys_vars['done']
        timestep_check = sys_vars['t'] == (self.env_spec['timestep_limit'] - 1)
        if done or timestep_check:
            self.update_target_model()
        super(FreezeDQN, self).update(sys_vars)
