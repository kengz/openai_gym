from rl.agent.dqn import DQN
from rl.util import logger
import math


class DDPGTF(DQN):

    '''
    The DDPG agent (algo), from https://arxiv.org/abs/1509.02971
    Code adapted from  https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
    '''

    def __init__(self, *args, **kwargs):
        # import only when needed to contain side-effects
        import numpy as np
        np.random.seed(1234)
        import tensorflow as tf
        import tflearn
        self.tf = tf
        self.tflearn = tflearn
        self.np = np
        self.tf.set_random_seed(1234)
        self.sess = tf.Session()
        self.action_bound = 2.0
        self.TAU = 0.001  # for target network updates
        self.lr_actor = 0.0001
        super(DDPGTF, self).__init__(*args, **kwargs)
        self.type = 'tensorflow'
        self.ep_avg_max_q = 0.0

    def compile(self, memory, optimizer, policy, preprocessor):
        self.optimizer = optimizer
        super(DDPGTF, self).compile(memory, self.optimizer, policy, preprocessor)

    def create_actor(self):
        inputs = self.tflearn.input_data(shape=[None, self.env_spec['state_dim']])
        net = self.tflearn.fully_connected(inputs, self.hidden_layers[0], activation=self.hidden_layers_activation)
        for i in range (1, len(self.hidden_layers)):
            net = self.tflearn.fully_connected(net, self.hidden_layers[i], activation=self.hidden_layers_activation)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = self.tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = self.tflearn.fully_connected(
            net, self.env_spec['action_dim'], activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = self.tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def build_actor_models(self):
        # Actor Network
        self.a_inputs, self.a_out, self.a_scaled_out = self.create_actor()
        self.a_network_params = self.tf.trainable_variables()

        # Target Network
        self.a_target_inputs, self.a_target_out, self.a_target_scaled_out = self.create_actor()
        self.a_target_network_params = self.tf.trainable_variables()[
            len(self.a_network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.a_update_target_network_params = \
            [self.a_target_network_params[i].assign(self.tf.multiply(self.a_network_params[i], self.TAU) +
                                                  self.tf.multiply(self.a_target_network_params[i], 1. - self.TAU))
                for i in range(len(self.a_target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = self.tf.placeholder(self.tf.float32, [None, self.env_spec['action_dim']])

        # Combine the gradients here
        self.actor_gradients = self.tf.gradients(
            self.a_scaled_out, self.a_network_params, -self.action_gradient)

        # Optimization Op
        self.a_optimize = self.tf.train.AdamOptimizer(self.lr_actor).\
            apply_gradients(zip(self.actor_gradients, self.a_network_params))

        self.a_num_trainable_vars = len(
            self.a_network_params) + len(self.a_target_network_params)

    def actor_train(self, inputs, a_gradient):
        self.sess.run(self.a_optimize, feed_dict={
            self.a_inputs: inputs,
            self.action_gradient: a_gradient
        })

    def actor_predict(self, inputs):
        return self.sess.run(self.a_scaled_out, feed_dict={
            self.a_inputs: inputs
        })

    def actor_predict_target(self, inputs):
        return self.sess.run(self.a_target_scaled_out, feed_dict={
            self.a_target_inputs: inputs
        })

    def actor_update_target_network(self):
        self.sess.run(self.a_update_target_network_params)

    def create_critic(self):
        inputs = self.tflearn.input_data(shape=[None, self.env_spec['state_dim']])
        action = self.tflearn.input_data(shape=[None, self.env_spec['action_dim']])
        h1_shape = self.hidden_layers[0] if len(self.hidden_layers) > 1 else math.floor(self.hidden_layers[0]  * 1.25)
        # print("H1 SHAPE: {}".format(h1_shape))
        net = self.tflearn.fully_connected(inputs, h1_shape, activation=self.hidden_layers_activation)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        h2_shape = self.hidden_layers[1] if len(self.hidden_layers) > 1 else self.hidden_layers[0] 
        # print("H2 SHAPE: {}".format(h2_shape))
        t1 = self.tflearn.fully_connected(net, h2_shape)
        t2 = self.tflearn.fully_connected(action, h2_shape)

        net = self.tflearn.activation(
            self.tf.matmul(net, t1.W) + self.tf.matmul(action, t2.W) + t2.b, activation=self.hidden_layers_activation)

        # Add remaining layers if they exist
        for i in range (2, len(self.hidden_layers)):
            net = self.tflearn.fully_connected(net, self.hidden_layers[i], activation=self.hidden_layers_activation)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = self.tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = self.tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def build_critic_models(self):
        #Critic Network
        self.c_inputs, self.c_action, self.c_out = self.create_critic()
        self.c_network_params = self.tf.trainable_variables()[self.a_num_trainable_vars:]

        # Target Network
        self.c_target_inputs, self.c_target_action, self.c_target_out = self.create_critic()
        self.c_target_network_params = self.tf.trainable_variables()[(len(self.c_network_params) + self.a_num_trainable_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.c_update_target_network_params = \
            [self.c_target_network_params[i].assign(self.tf.multiply(self.c_network_params[i], self.TAU) + self.tf.multiply(self.c_target_network_params[i], 1. - self.TAU))
                for i in range(len(self.c_target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = self.tf.placeholder(self.tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = self.tflearn.mean_square(self.predicted_q_value, self.c_out)
        self.c_optimize = self.tf.train.AdamOptimizer(
            self.lr).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = self.tf.gradients(self.c_out, self.c_action)

    def critic_train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.c_out, self.c_optimize], feed_dict={
            self.c_inputs: inputs,
            self.c_action: action,
            self.predicted_q_value: predicted_q_value
        })

    def critic_predict(self, inputs, action):
        return self.sess.run(self.c_out, feed_dict={
            self.c_inputs: inputs,
            self.c_action: action
        })

    def critic_predict_target(self, inputs, action):
        return self.sess.run(self.c_target_out, feed_dict={
            self.c_target_inputs: inputs,
            self.c_target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.c_inputs: inputs,
            self.c_action: actions
        })

    def critic_update_target_network(self):
        self.sess.run(self.c_update_target_network_params)

    def build_model(self):
        self.build_actor_models()
        self.build_critic_models()

    def compile_model(self):
        self.sess.run(self.tf.global_variables_initializer())
        logger.info("Tensorflow variables initializaed")

    def update(self, sys_vars):
        '''Agent update apart from training the Q function'''
        self.policy.update(sys_vars)
        self.update_n_epoch(sys_vars)

    def train_critic(self, minibatch):
        '''update critic network using K-mean loss'''
        actions = self.actor_predict_target(minibatch['next_states'])
        Q_prime = self.critic_predict_target(minibatch['next_states'], actions)
        y_i = []
        for k in range(self.batch_size):
            if minibatch['terminals'][k]:
                y_i.append(minibatch['rewards'][k])
            else:
                y_i.append(minibatch['rewards'][k] + self.gamma * Q_prime[k])
        y_i = self.np.reshape(y_i, (self.batch_size, 1))
        predicted_q_value, _ = self.critic_train(minibatch['states'], actions, y_i)
        # self.ep_avg_max_q += self.np.amax(predicted_q_value)
        critic_loss = 0
        return critic_loss

    def train_actor(self, minibatch):
        '''update actor network using sampled gradient'''
        actions = self.actor_predict(minibatch['states'])
        grads = self.action_gradients(minibatch['states'], actions)
        self.actor_train(minibatch['states'], grads[0])
        actor_loss = 0
        return actor_loss

    def train_target_networks(self):
        '''update both target networks'''
        self.actor_update_target_network()
        self.critic_update_target_network()

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)
        # print(minibatch['actions'])
        critic_loss = self.train_critic(minibatch)
        actor_loss = self.train_actor(minibatch)
        self.train_target_networks()

        loss = critic_loss + actor_loss
        return loss
