import numpy as np
from rl.agent.base_agent import Agent
from rl.util import logger, log_self


class DQN(Agent):

    '''
    The base class of DQNs, with the core methods
    The simplest deep Q network,
    with epsilon-greedy method and
    Bellman equation for value, using neural net.
    '''

    def __init__(self, env_spec,
                 train_per_n_new_exp=1,
                 gamma=0.95, lr=0.1,
                 epi_change_lr=None,
                 batch_size=16, n_epoch=5, hidden_layers_shape=None,
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear',
                 auto_architecture=False,
                 num_hidden_layers=3,
                 size_first_hidden_layer=256,
                 num_initial_channels=16,
                 **kwargs):  # absorb generic param without breaking
        # import only when needed to contain side-effects
        from keras.layers.core import Dense
        from keras.models import Sequential, load_model
        self.Dense = Dense
        self.Sequential = Sequential
        self.load_model = load_model

        super(DQN, self).__init__(env_spec)

        self.train_per_n_new_exp = train_per_n_new_exp
        self.gamma = gamma
        self.lr = lr
        self.epi_change_lr = epi_change_lr
        self.batch_size = batch_size
        self.n_epoch = 1
        self.final_n_epoch = n_epoch
        self.hidden_layers = hidden_layers_shape or [4]
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation = output_layer_activation
        self.clip_val = 10000
        self.auto_architecture = auto_architecture
        self.num_hidden_layers = num_hidden_layers
        self.size_first_hidden_layer = size_first_hidden_layer
        self.num_initial_channels = num_initial_channels
        log_self(self)
        self.build_model()

    def build_hidden_layers(self, model):
        '''
        build the hidden layers into model using parameter self.hidden_layers
        '''

        # Auto architecture infers the size of the hidden layers from the size
        # of the first layer. Each successive hidden layer is half the size of the
        # previous layer
        # Enables hyperparameter optimization over network architecture
        if self.auto_architecture:
            curr_layer_size = self.size_first_hidden_layer
            model.add(self.Dense(curr_layer_size,
                                 input_shape=(self.env_spec['state_dim'],),
                                 activation=self.hidden_layers_activation,
                                 init='lecun_uniform'))

            curr_layer_size = int(curr_layer_size / 2)
            for i in range(1, self.num_hidden_layers):
                model.add(self.Dense(curr_layer_size,
                                     init='lecun_uniform',
                                     activation=self.hidden_layers_activation))
                curr_layer_size = int(curr_layer_size / 2)

        else:
            model.add(self.Dense(self.hidden_layers[0],
                                 input_shape=(self.env_spec['state_dim'],),
                                 activation=self.hidden_layers_activation,
                                 init='lecun_uniform'))
            # inner hidden layer: no specification of input shape
            if (len(self.hidden_layers) > 1):
                for i in range(1, len(self.hidden_layers)):
                    model.add(self.Dense(
                        self.hidden_layers[i],
                        init='lecun_uniform',
                        activation=self.hidden_layers_activation))

        return model

    def build_model(self):
        model = self.Sequential()
        self.build_hidden_layers(model)
        model.add(self.Dense(self.env_spec['action_dim'],
                             init='lecun_uniform',
                             activation=self.output_layer_activation))
        logger.info("Model summary")
        model.summary()
        self.model = model

        logger.info("Model built")
        return self.model

    def compile_model(self):
        self.model.compile(
            loss='mean_squared_error',
            optimizer=self.optimizer.keras_optimizer)
        logger.info("Model compiled")

    def recompile_model(self, sys_vars):
        '''
        Option to change model optimizer settings
        Currently only used for changing the learning rate
        Compiling does not affect the model weights
        '''
        if self.epi_change_lr is not None:
            if (sys_vars['epi'] == self.epi_change_lr and
                    sys_vars['t'] == 0):
                self.lr = self.lr / 10.0
                self.optimizer.change_optim_param(**{'lr': self.lr})
                self.model.compile(
                    loss='mean_squared_error',
                    optimizer=self.optimizer.keras_optimizer)
                logger.info('Model recompiled with new settings: '
                            'Learning rate: {}'.format(self.lr))
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
        Q_states = np.clip(self.model.predict(minibatch['states']),
                           -self.clip_val, self.clip_val)
        Q_next_states = np.clip(self.model.predict(minibatch['next_states']),
                                -self.clip_val, self.clip_val)
        Q_next_states_max = np.amax(Q_next_states, axis=1)
        return (Q_states, Q_next_states, Q_next_states_max)

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
        (Q_states, _states, Q_next_states_max) = self.compute_Q_states(
            minibatch)
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
