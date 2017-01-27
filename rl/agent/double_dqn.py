import numpy as np
from rl.agent.dqn import DQN
from rl.util import logger
from keras.models import Sequential


class DoubleDQN(DQN):

    '''
    The base class of double DQNs
    '''

    def build_model(self):
        super(DoubleDQN, self).build_model()

        model2 = Sequential.from_config(self.model.get_config())
        logger.info("Model 2 summary")
        model2.summary()
        self.model2 = model2

        self.model2.compile(
            loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Models built and compiled")
        return self.model, self.model2

    def compute_Q_states(self, minibatch):
        clip_val = 10000
        Q_states = np.clip(
            self.model.predict(minibatch['states']), -clip_val, clip_val)
        # Different from (single) dqn: Select max using model 2
        Q_next_states_select = np.clip(
            self.model2.predict(minibatch['next_states']), -clip_val, clip_val)
        Q_next_states_max_ind = np.argmax(Q_next_states_select, axis=1)
        # if more than one max, pick 1st
        if (Q_next_states_max_ind.shape[0] > 1):
            Q_next_states_max_ind = Q_next_states_max_ind[0]

        # same as dqn again, but use Q_next_states_max_ind above
        Q_next_states = np.clip(
            self.model.predict(minibatch['next_states']), -clip_val, clip_val)
        Q_next_states_max = Q_next_states[:, Q_next_states_max_ind]
        return (Q_states, Q_next_states_max)

    def train_an_epoch(self):
         # Switch model 1 and model 2
        temp = self.model
        self.model = self.model2
        self.model2 = temp
        return super(DoubleDQN, self).train_an_epoch()
