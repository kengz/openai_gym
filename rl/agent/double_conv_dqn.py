from rl.agent.conv_dqn import ConvDQN
import numpy as np
from rl.util import logger


class DoubleConvDQN(ConvDQN):

    '''
    The base class of double convolutional DQNs
    '''

    def build_model(self):
        super(DoubleConvDQN, self).build_model()

        model2 = self.Sequential.from_config(self.model.get_config())
        logger.info("Model 2 summary")
        model2.summary()
        self.model2 = model2

        logger.info("Models 1 and 2 built")
        return self.model, self.model2

    def compile_model(self):
        self.model.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer.keras_optimizer)
        self.model2.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer.keras_optimizer)
        logger.info("Models 1 and 2 compiled")

    def compute_Q_states(self, minibatch):
        (Q_states, Q_next_states_select, _max) = super(
            DoubleConvDQN, self).compute_Q_states(minibatch)
        # Different from (single) dqn: Select max using model 2
        Q_next_states_max_ind = np.argmax(Q_next_states_select, axis=1)
        # same as dqn again, but use Q_next_states_max_ind above
        Q_next_states = np.clip(
            self.model2.predict(minibatch['next_states']),
            -self.clip_val, self.clip_val)
        rows = np.arange(Q_next_states_max_ind.shape[0])
        Q_next_states_max = Q_next_states[rows, Q_next_states_max_ind]

        return (Q_states, Q_next_states, Q_next_states_max)

    def switch_models(self):
         # Switch model 1 and model 2
        temp = self.model
        self.model = self.model2
        self.model2 = temp

    def train_an_epoch(self):
        self.switch_models()
        return super(DoubleConvDQN, self).train_an_epoch()
