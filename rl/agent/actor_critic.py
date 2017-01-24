import os
import numpy as np
from rl.agent.dqn import DQN
from rl.util import logger, log_self
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.objectives import mean_squared_error
from keras.constraints import maxnorm
from keras import backend as K
from theano.tensor import nnet

class ActorCritic(DQN):

    def __init__(self, env_spec,
                 train_per_n_new_exp=1,
                 gamma=0.99, learning_rate=0.01,
                 epi_change_learning_rate=None,
                 batch_size=16, n_epoch=5, hidden_layers_shape=[4],
                 hidden_layers_activation='sigmoid',
                 output_layer_activation='linear',
                 **kwargs):  # absorb generic param without breaking
        super(ActorCritic, self).__init__(env_spec,
                                                     train_per_n_new_exp,
                                                     gamma, learning_rate,
                                                     epi_change_learning_rate,
                                                     batch_size, n_epoch, hidden_layers_shape,
                                                     hidden_layers_activation,
                                                     output_layer_activation,
                                                     **kwargs)
        self.actor_opt = None
        self.actor_loss = None
        self.actor_loss_set = False
        self.critic_opt = None
        self.critic_loss = None
        self.curr_actions = np.zeros(self.env_spec['action_dim'])
        self.kwargs = kwargs
        # Doesn't work with JSON
        # log_self(self)
        self.build_ac_model()

    def build_ac_model(self):
        self.actor = DQN(self.env_spec,
            train_per_n_new_exp=self.train_per_n_new_exp,
            gamma=self.gamma, 
            learning_rate=self.learning_rate,
            epi_change_learning_rate=self.epi_change_learning_rate,
            batch_size=self.batch_size, 
            n_epoch=self.n_epoch, 
            hidden_layers_shape=self.hidden_layers,
            hidden_layers_activation=self.hidden_layers_activation,
            output_layer_activation='linear',
             **self.kwargs)
        self.critic = DQN(self.env_spec,
           train_per_n_new_exp=self.train_per_n_new_exp,
            gamma=self.gamma, 
            learning_rate=self.learning_rate,
            epi_change_learning_rate=self.epi_change_learning_rate,
            batch_size=self.batch_size, 
            n_epoch=self.n_epoch, 
            hidden_layers_shape=self.hidden_layers,
            hidden_layers_activation=self.hidden_layers_activation,
            output_layer_activation='linear',
             **self.kwargs)
        self.critic.model.pop()
        self.critic.model.add(Dense(1,
                    init='lecun_uniform',
                    activation=self.output_layer_activation,
                    W_constraint=maxnorm(3)))
        print(self.critic.model.summary())
        self.compile_models()
        logger.info("Actor and critic models built and compiled")
        return self.actor.model, self.critic.model

    def custom_objective(self, error, y_pred):
        y_pred_prob = nnet.nnet.softmax(y_pred)
        loss = - K.log(y_pred_prob) * self.curr_actions * error
        return loss

    def compile_models(self):
        self.actor_opt  = SGD(lr=self.learning_rate)
        self.critic_opt  = SGD(lr=self.learning_rate)
        self.actor_loss = "mean_squared_error"
        self.critic_loss = "mean_squared_error"
        self.actor.model.compile(loss=self.actor_loss, optimizer=self.actor_opt)
        self.critic.model.compile(loss=self.critic_loss, optimizer=self.critic_opt)

    def add_custom_objective(self):
        self.actor_loss = self.custom_objective
        self.actor.model.compile(loss=self.actor_loss, optimizer=self.actor_opt)

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
                self.actor_opt = SGD(lr=self.learning_rate)
                self.critic_opt = SGD(lr=self.learning_rate)
                self.actor.model.compile(loss=self.actor_loss, optimizer=self.actor_opt)
                self.critic.model.compile(loss=self.critic_loss, optimizer=self.critic_opt)
                logger.info('Actor and critic models recompiled with new settings: '
                            'Learning rate: {}'.format(self.learning_rate))
            return self.actor.model, self.critic.model

    def select_action(self, state):
        # Assumes policy is appropriate to this agent
        return self.policy.select_action(state)

    def compute_action_vals(self, minibatch):
        return self.actor.model.predict(minibatch['states']) 

    def train_an_epoch(self):
        if not self.actor_loss_set:
            self.add_custom_objective()
            self.actor_loss_set = True
        minibatch = self.memory.rand_minibatch(self.batch_size)
        self.curr_actions = minibatch['actions']
        curr_state_vals = self.critic.model.predict(minibatch['states'])
        next_state_vals = self.critic.model.predict(minibatch['next_states'])
        curr_targets = minibatch['rewards'] .reshape((minibatch['rewards'].shape[0], 1))+ self.gamma * next_state_vals
        error = curr_targets - curr_state_vals
        # Broadcast error to match model output dim
        error = np.broadcast_to(error, (error.shape[0], self.env_spec['action_dim']))
        loss_critic = self.critic.model.train_on_batch(minibatch['states'], curr_targets)
        loss_actor = self.actor.model.train_on_batch(minibatch['states'], error)
        return loss_critic + loss_actor

    def save(self, model_path, global_step=None):
        logger.info('Saving model checkpoint')
        self.critic.model.save_weights(os.path.join(model_path, "_critic"))
        self.actor.model.save_weights(os.path.join(model_path, "_actor"))

    def restore(self, model_path):
        self.critic.model.load_weights(os.path.join(model_path, "_critic"), by_name=False)
        self.actor.model.load_weights(os.path.join(model_path, "_actor"), by_name=False)



