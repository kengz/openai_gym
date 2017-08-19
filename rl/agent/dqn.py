import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from rl.agent.base_agent import Agent
from rl.util import logger, log_self

NumpyType = Any

def get_activation_fn(name):
    def linear(x):
        return x

    return { "relu": F.relu, "sigmoid": F.sigmoid, "linear": linear }[name]

def verify_contains_subset(d : Dict[Any, Any], keys : List[Any]):
    assert isinstance(d, dict)
    obtained = set(d.keys())
    expected = set(keys)
    if not expected.issubset(obtained):
        msg = "Given a dictionary with keys {} ; but expected at least the following keys {}"
        msg = msg.format(
                ",".join(str(key) for key in obtained),
                ",".join(str(key) for key in expected))
        raise AssertionError(msg)


def parse_minibatch_to_torch(
        minibatch: Dict[str, NumpyType]
    ) -> Dict[str, autograd.Variable]:
    '''
    Args:
        minibatch: A minibatch dictionary obtained from replay memory.
    Returns:
        A dictionary with the same keys that maps to torch variables.
    '''

    verify_contains_subset(minibatch, [
            'states',
            'rewards',
            'next_states',
            'terminals',
            'actions',
            ])
    return {
        k: autograd.Variable(torch.from_numpy(v).float(), requires_grad=True)
        for k, v in minibatch.items()
    }

def build_hidden_layers(dqn) -> Tuple[List[nn.Linear], List[int]]:
    '''
    Args:
        dqn: A DQN object populated with the relevant fields.
    Returns:
        A pair of lists with the transformation matrix and the layer output
        dimensions rrepsctively. Both lists have the same length.
    '''
    # Auto architecture infers the size of the hidden layers from the size
    # of the first layer. Each successive hidden layer is half the size of the
    # previous layer
    # Enables hyperparameter optimization over network architecture
    layers = []
    dims = []

    if dqn.auto_architecture:
        curr_layer_size = dqn.first_hidden_layer_size
        dims.append(cur_layer_size)
        layers.append(nn.Linear(dqn.env_spec['state_dim'], cur_layer_size))

        prev_layer_size = curr_layer_size
        curr_layer_size = int(curr_layer_size / 2)
        for i in range(1, dqn.num_hidden_layers):
            dims.append(cur_layer_size)
            layers.append(nn.Linear(prev_layer_size, curr_layer_size))
            prev_layer_size = curr_layer_size
            curr_layer_size = int(curr_layer_size / 2)

    else:
        dims = list(dqn.hidden_layers)
        layers.append(
            nn.Linear(dqn.env_spec['state_dim'], dqn.hidden_layers[0]))

        # inner hidden layer: no specification of input shape
        for i in range(1, len(dqn.hidden_layers)):
            layers.append(
                    nn.Linear(
                        dqn.hidden_layers[i - 1],
                        dqn.hidden_layers[i]))

    for layer in layers:
        tensor = layer.weight.data
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform(
                tensor, a=-math.sqrt(3 / fan_in), b=math.sqrt(3 / fan_in))
        layer.bias.data.fill_(0.)

    return layers, dims


class Net(nn.Module):

    def __init__(self, dqn):
        '''
        Args:
            dqn: An instance of the DQN object with the properties to define
                 the specifciation of the net.
        '''
        super().__init__()
        hidden_layers, hidden_layer_dims = build_hidden_layers(dqn)

        self._hidden_layers_activation = get_activation_fn(
                dqn.hidden_layers_activation)
        self._output_layer_activation = get_activation_fn(
                dqn.output_layer_activation)
        self._hidden_layers = nn.ModuleList(hidden_layers)
        self._output_layer = nn.Linear(
                hidden_layer_dims[-1], dqn.env_spec['action_dim'])

    def forward(self, x):
        for layer in self._hidden_layers:
            x = self._hidden_layers_activation(layer(x))
        return self._output_layer_activation(self._output_layer(x))


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
                 batch_size=16, n_epoch=5, hidden_layers=None,
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear',
                 auto_architecture=False,
                 num_hidden_layers=3,
                 first_hidden_layer_size=256,
                 num_initial_channels=16,
                 **kwargs):  # absorb generic param without breaking
        super(DQN, self).__init__(env_spec)

        self.train_per_n_new_exp = train_per_n_new_exp
        self.gamma = gamma
        self.lr = lr
        self.epi_change_lr = epi_change_lr
        self.batch_size = batch_size
        self.n_epoch = 1
        self.final_n_epoch = n_epoch
        self.hidden_layers = hidden_layers or [4]
        self.hidden_layers_activation = hidden_layers_activation
        self.output_layer_activation = output_layer_activation
        self.clip_val = 10000
        self.auto_architecture = auto_architecture
        self.num_hidden_layers = num_hidden_layers
        self.first_hidden_layer_size = first_hidden_layer_size
        self.num_initial_channels = num_initial_channels
        log_self(self)
        self.build_model()

    def build_model(self):
        self.model = Net(self)
        return self.model

    def compile_model(self):
        self._loss_fn = torch.nn.MSELoss(size_average=True)
        self.torch_optimizer = self.optimizer.torch_optimizer(
                self.model.parameters())
        self.torch_optimizer.zero_grad()
        logger.info("Model compiled")

    def recompile_model(self, sys_vars):
        '''
        Option to change model optimizer settings
        Currently only used for changing the learning rate
        Compiling does not affect the model weights
        '''
        return self.model
        if self.epi_change_lr is not None:
            if (sys_vars['epi'] == self.epi_change_lr and
                    sys_vars['t'] == 0):
                self.lr = self.lr / 10.0
                self.optimizer.change_optim_param(**{'lr': self.lr})
                self.model.compile(
                    loss='mse',
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
        Q_states = self.model(minibatch['states']).clamp(
                min=-self.clip_val, max=self.clip_val)
        Q_next_states = self.model(minibatch['next_states']).clamp(
                min=-self.clip_val, max=self.clip_val)
        Q_next_states_max, _ = torch.max(Q_next_states, dim=1)
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
        minibatch = parse_minibatch_to_torch(
            self.memory.rand_minibatch(self.batch_size))
        (Q_states, _states, Q_next_states_max) = \
                self.compute_Q_states(minibatch)
        Q_targets = self.compute_Q_targets(
                minibatch, Q_states, Q_next_states_max)

        self.torch_optimizer.zero_grad()
        loss = self._loss_fn(Q_states, autograd.Variable(Q_targets.data))
        loss.backward()
        self.torch_optimizer.step()

        errors = abs(torch.sum(Q_states - Q_targets, dim=1).data.numpy())
        assert Q_targets.size() == (
            self.batch_size, self.env_spec['action_dim'])
        assert errors.shape == (self.batch_size, )
        self.memory.update(errors)
        return loss.data.numpy()

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
        torch.save(self.model.state_dict(), model_path)

    def restore(self, model_path):
        logger.info('Loading model checkpoint')
        self.model = torch.load(model_path)
