import numpy as np
from rl.agent.base_agent import Agent
from rl.util import logger, log_self
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


class DQN(Agent):

    '''
    The base class of DQNs, with the core methods
    The simplest deep Q network,
    with epsilon-greedy method and
    Bellman equation for value, using neural net.
    '''

    def __init__(self, env_spec,
                 train_per_n_new_exp=1,
                 gamma=0.95, learning_rate=0.1,
                 epi_change_learning_rate=None,
                 batch_size=16, n_epoch=5, hidden_layers_shape=[4],
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear',
                 **kwargs):  # absorb generic param without breaking
        super(DQN, self).__init__(env_spec)

        self.train_per_n_new_exp = train_per_n_new_exp
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epi_change_learning_rate = epi_change_learning_rate
        self.batch_size = batch_size
        self.n_epoch = 1
        self.final_n_epoch = n_epoch
        self.hidden_layers = hidden_layers_shape
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation = output_layer_activation
        log_self(self)
        self.optimizer = None
        self.build_model()

    def build_hidden_layers(self, model):
        '''
        build the hidden layers into model using parameter self.hidden_layers
        '''
        model.add(Dense(self.hidden_layers[0],
                        input_shape=(self.env_spec['state_dim'],),
                        activation=self.hidden_layers_activation,
                        init='lecun_uniform'))

        # inner hidden layer: no specification of input shape
        if (len(self.hidden_layers) > 1):
            for i in range(1, len(self.hidden_layers)):
                model.add(Dense(self.hidden_layers[i],
                                init='lecun_uniform',
                                activation=self.hidden_layers_activation))

        return model

    def build_optimizer(self):
        self.optimizer = SGD(lr=self.learning_rate)

    def build_model(self):
        model = Sequential()
        self.build_hidden_layers(model)
        model.add(Dense(self.env_spec['action_dim'],
                        init='lecun_uniform',
                        activation=self.output_layer_activation))
        logger.info("Model summary")
        model.summary()
        self.model = model

        self.build_optimizer()
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        logger.info("Model built and compiled")
        return self.model

    def recompile_model(self, sys_vars):
        '''
        Option to change model optimizer settings
        Currently only used for changing the learning rate
        Compiling does not affect the model weights
        '''
        if self.epi_change_learning_rate is not None:
            if (sys_vars['epi'] == self.epi_change_learning_rate and
                    sys_vars['t'] == 0):
                self.learning_rate = self.learning_rate / 10.0
                self.build_optimizer()
                self.model.compile(
                    loss='mean_squared_error', optimizer=self.optimizer)
                logger.info('Model recompiled with new settings: '
                            'Learning rate: {}'.format(self.learning_rate))
        return self.model

    def update_n_epoch(self, sys_vars):
        '''
        Increase epochs at the beginning of each session,
        for training for later episodes,
        once it has more experience
        Best so far, increment num epochs every 2 up to a max of 5
        '''
        if (self.n_epoch < self.final_n_epoch and
                sys_vars['t'] == 0 and
                sys_vars['epi'] % 2 == 0):
            self.n_epoch += 1
        return self.n_epoch

    def select_action(self, state):
        '''epsilon-greedy method'''
        return self.policy.select_action(state)

    def update(self, sys_vars):
        '''
        Agent update apart from training the Q function
        '''
        self.policy.update(sys_vars)
        self.update_n_epoch(sys_vars)
        self.recompile_model(sys_vars)

    def to_train(self, sys_vars):
        '''
        return boolean condition if agent should train
        get n NEW experiences before training model
        '''
        t = sys_vars['t']
        done = sys_vars['done']
        timestep_limit = self.env_spec['timestep_limit']
        return (t > 0) and bool(
            t % self.train_per_n_new_exp == 0 or
            t == (timestep_limit-1) or
            done)

    def compute_Q_states(self, minibatch):
        # note the computed values below are batched in array
        clip_val = 10000
        Q_states = np.clip(
            self.model.predict(minibatch['states']), -clip_val, clip_val)
        Q_next_states = np.clip(
            self.model.predict(minibatch['next_states']), -clip_val, clip_val)
        Q_next_states_max = np.amax(Q_next_states, axis=1)
        return (Q_states, Q_next_states_max)

    def compute_Q_targets(self, minibatch, Q_states, Q_next_states_max):
        # make future reward 0 if exp is terminal
        Q_targets_a = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_next_states_max
        # set batch Q_targets of a as above, the rest as is
        # minibatch['actions'] is one-hot encoded
        Q_targets = minibatch['actions'] * Q_targets_a[:, np.newaxis] + \
            (1 - minibatch['actions']) * Q_states
        return Q_targets

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)
        (Q_states, Q_next_states_max) = self.compute_Q_states(minibatch)
        Q_targets = self.compute_Q_targets(
            minibatch, Q_states, Q_next_states_max)

        loss = self.model.train_on_batch(minibatch['states'], Q_targets)
        return loss

    def train(self, sys_vars):
        '''
        Training is for the Q function (NN) only
        otherwise (e.g. policy) see self.update()
        step 1,2,3,4 of algo.
        '''
        loss_total = 0
        for _epoch in range(self.n_epoch):
            loss = self.train_an_epoch()
            loss_total += loss
        avg_loss = loss_total / self.n_epoch
        sys_vars['loss'].append(avg_loss)
        return avg_loss

    def save(self, model_path, global_step=None):
        logger.info('Saving model checkpoint')
        self.model.save_weights(model_path)

    def restore(self, model_path):
        self.model.load_weights(model_path, by_name=False)
