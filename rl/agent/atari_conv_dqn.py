from rl.agent.dqn import DQN
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
from keras import backend as K
K.set_image_dim_ordering('tf')


class ConvDQN(DQN):

    def __init__(self, *args, **kwargs):
        super(ConvDQN, self).__init__(*args, **kwargs)

    def build_hidden_layers(self, model):
        '''
        build the hidden layers into model using parameter self.hidden_layers
        '''
        model.add(
            Convolution2D(
                self.hidden_layers[0][0],
                self.hidden_layers[0][1],
                self.hidden_layers[0][2],
                subsample=self.hidden_layers[0][3],
                input_shape=(self.env_spec['state_dim']),
                activation=self.hidden_layers_activation,
                init='lecun_uniform'))

        if (len(self.hidden_layers) > 1):
            for i in range(1, len(self.hidden_layers)):
                model.add(
                    Convolution2D(
                        self.hidden_layers[i][0],
                        self.hidden_layers[i][1],
                        self.hidden_layers[i][2],
                        subsample=self.hidden_layers[i][3],
                        activation=self.hidden_layers_activation,
                        init='lecun_uniform'))

        model.add(Flatten())
        model.add(Dense(256,
                        init='lecun_uniform',
                        activation=self.hidden_layers_activation))

        return model

    def build_optimizer(self):
        '''Optimizer is RMSprop, default learning rate used'''
        self.optimizer = RMSprop()
