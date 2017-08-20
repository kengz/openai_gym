from typing import List, Tuple
import collections
import math

from torch import nn

from rl import torch_utils
from rl.agent import dqn

ConvSpec = collections.namedtuple(
        'ConvSpec', ['channels', 'kernel', 'stride'])

def auto_architect_hidden_layers(conv_dqn):
    conv_specs = []
    stride = conv_dqn.stride
    kernel = conv_dqn.kernel
    output_channels = conv_dqn.num_initial_channels
    cols, rows, input_channels = conv_dqn.env_spec['state_dim']

    conv_specs.append(
            ConvSpec(
                channels=output_channels,
                kernel=kernel,
                stride=stride))
    for i in range(1, conv_dqn.num_hidden_layers):
        output_channels = output_channels * 2
        cols = math.ceil(math.floor(cols - kernel - 1) / stride[0]) + 1
        rows = math.ceil(math.floor(rows - kernel - 1) / stride[1]) + 1
        if cols > 5 and rows > 5:
            spec = ConvSpec(
                    channels=output_channels,
                    kernel=kernel,
                    stride=stride)
            conv_specs.append(spec)
        else:
            # stop addition of too many layers
            # and from breakage by cols, rows growing to 0
            break

    return conv_specs


def build_hidden_layers(conv_dqn) -> Tuple[List[nn.Conv2d], List[nn.Linear]]:
    '''
    build the hidden layers into model using parameter self.hidden_layers
    Auto architecture infers the size of the hidden layers from the number
    of channels in the first hidden layer and number of layers
    With each successive layer the number of channels is doubled
    Kernel size is fixed at 4, and stride at (2, 2)
    No new layers are added if the cols or rows have dim <= 5
    Enables hyperparameter optimization over network architecture
    '''
    if conv_dqn.auto_architecture:
        print("Conv_dqn.num_initial_channels", conv_dqn.num_initial_channels)
        print("Conv_dqn.num_hidden_layers", conv_dqn.num_hidden_layers)
        conv_specs = auto_architect_hidden_layers(conv_dqn)
    else:
        conv_specs = []
        for i in range(len(conv_dqn.hidden_layers)):
            kernel = (conv_dqn.hidden_layers[i][1],
                      conv_dqn.hidden_layers[i][2])
            conv_specs.append(ConvSpec(
                channels=conv_dqn.hidden_layers[i][0],
                kernel=kernel,
                stride=tuple(conv_dqn.hidden_layers[i][3])))

    conv_layers = []
    cols, rows, input_channels = conv_dqn.env_spec['state_dim']

    for i in range(0, len(conv_specs)):
        if i != 0:
            input_channels = conv_specs[i - 1].channels
        kernel = conv_specs[i].kernel
        stride = conv_specs[i].stride
        conv_layer = nn.Conv2d(
                input_channels,
                conv_specs[i].channels,
                conv_specs[i].kernel,
                stride=conv_specs[i].stride)
        if isinstance(kernel, tuple):
            row_kernel, col_kernel = kernel
        elif isinstance(kernel, int):
            col_kernel = kernel
            row_kernel = kernel
        else:
            assert false
        cols = math.ceil(math.floor(cols - col_kernel - 1) / stride[0]) + 1
        rows = math.ceil(math.floor(rows - row_kernel - 1) / stride[1]) + 1
        conv_layers.append(conv_layer)

    fc_layers = [nn.Linear(rows * cols * conv_layers[-1].out_channels, 256)]

    for layer in fc_layers + conv_layers:
        dqn.lecun_init(layer.weight.data)
        layer.bias.data.fill_(0.)

    return (conv_layers, fc_layers)

class ConvNet(nn.Module):

    def __init__(self, conv_dqn):
        super().__init__()
        self._conv_layers, self._fc_hidden_layers = \
                build_hidden_layers(conv_dqn)
        self._output_layer = nn.Linear(
                self._fc_hidden_layers[-1].weight.size()[1],
                conv_dqn.env_spec['action_dim'])
        self._hidden_layers_activation = dqn.get_activation_fn(
                conv_dqn.hidden_layers_activation)
        self._output_layer_activation = dqn.get_activation_fn(
                conv_dqn.hidden_layers_activation)

    def forward(self, x):
        for conv_layer in self._conv_layers:
            print("input size =", x.size())
            print("conv layer =", conv_layer)
            x = self._hidden_layers_activation(conv_layer(x))
        x = x.view(-1, self._fc_hidden_layers.weight.size()[0])

        for hidden_layer in self._fc_hidden_layers:
            x = self._hidden_layers_activation(hidden_layer(x))
        return self._output_layer_activation(self._output_layer(x))


class ConvDQN(dqn.DQN):

    def __init__(self, *args, **kwargs):
        self.kernel = 4
        self.stride = (2, 2)
        super(ConvDQN, self).__init__(*args, **kwargs)

    def build_model(self):
        self.model = ConvNet(self)
        torch_utils.maybe_cuda(self.model)
        return self.model
