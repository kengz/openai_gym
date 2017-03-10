from rl.agent.dqn import DQN
import math

class ConvDQN(DQN):

    def __init__(self, *args, **kwargs):
        from keras.layers.core import Dense, Flatten
        from keras.layers.convolutional import Convolution2D
        from keras import backend as K
        if K.backend() == 'theano':
            K.set_image_dim_ordering('tf')
        self.Dense = Dense
        self.Flatten = Flatten
        self.Convolution2D = Convolution2D
        self.kernel = 4
        self.stride = (2, 2)
        super(ConvDQN, self).__init__(*args, **kwargs)

    def build_hidden_layers(self, model):
        '''
        build the hidden layers into model using parameter self.hidden_layers
        '''
        # Auto architecture infers the size of the hidden layers from the number
        # of channels in the first hidden layer and number of layers
        # With each successive layer the number of channels is doubled
        # Kernel size is fixed at 4, and stride at (2, 2)
        # No new layers are added if the cols or rows have dim <= 5
        # Enables hyperparameter optimization over network architecture
        if self.auto_architecture:
            num_channels = self.num_initial_channels
            cols = self.env_spec['state_dim'][0]
            rows = self.env_spec['state_dim'][1]
            model.add(
                self.Convolution2D(
                    num_channels,
                    self.kernel,
                    self.kernel,
                    subsample=self.stride,
                    input_shape=self.env_spec['state_dim'],
                    activation=self.hidden_layers_activation,
                    # border_mode='same',
                    init='lecun_uniform'))
            num_channels *= 2
            cols = math.ceil(math.floor(cols  - self.kernel -1) / self.stride[0]) + 1
            rows = math.ceil(math.floor(rows - self.kernel -1) / self.stride[1]) + 1
            for i in range(1, self.num_hidden_layers):
                if  cols > 5 and rows > 5:
                    model.add(
                        self.Convolution2D(
                        num_channels,
                        self.kernel,
                        self.kernel,
                        subsample=self.stride,
                        activation=self.hidden_layers_activation,
                        # border_mode='same',
                        init='lecun_uniform'))
                    num_channels *= 2
                    cols = math.ceil(math.floor(cols  - self.kernel -1) / self.stride[0]) + 1
                    rows = math.ceil(math.floor(rows - self.kernel -1) / self.stride[1]) + 1

        else:
            model.add(
                self.Convolution2D(
                    self.hidden_layers[0][0],
                    self.hidden_layers[0][1],
                    self.hidden_layers[0][2],
                    subsample=self.hidden_layers[0][3],
                    input_shape=self.env_spec['state_dim'],
                    activation=self.hidden_layers_activation,
                    # border_mode='same',
                    init='lecun_uniform'))

            if (len(self.hidden_layers) > 1):
                for i in range(1, len(self.hidden_layers)):
                    model.add(
                        self.Convolution2D(
                            self.hidden_layers[i][0],
                            self.hidden_layers[i][1],
                            self.hidden_layers[i][2],
                            subsample=self.hidden_layers[i][3],
                            activation=self.hidden_layers_activation,
                            # border_mode='same',
                            init='lecun_uniform'))

        model.add(self.Flatten())
        model.add(self.Dense(256,
                             init='lecun_uniform',
                             activation=self.hidden_layers_activation))

        return model
