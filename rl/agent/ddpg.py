from rl.agent.dqn import DQN
from rl.util import logger, clone_model, clone_optimizer, ddpg_weight_init, tanh2, normal_02
import math



class DDPG(DQN):

    '''
    The DDPG agent (algo), from https://arxiv.org/abs/1509.02971
    reference: https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
    https://github.com/matthiasplappert/keras-rl
    '''

    def __init__(self, *args, **kwargs):
        # import only when needed to contain side-effects
        import numpy as np
        np.random.seed(1234)
        from keras.layers import Dense, Merge
        from keras.models import Sequential
        from keras import backend as K
        from keras.initializations import uniform
        self.Dense = Dense
        self.Merge = Merge
        self.Sequential = Sequential
        self.uniform = uniform
        self.K = K
        import tensorflow as tf
        self.tf = tf
        self.tf.set_random_seed(1234)
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.TAU = 0.001  # for target network updates
        super(DDPG, self).__init__(*args, **kwargs)
        self.lr_actor = 0.0001
        # print("ACTOR LEARNING RATE")
        # print(self.lr_actor)

    def compile(self, memory, optimizer, policy, preprocessor):
        # override to make 4 optimizers
        self.optimizer = optimizer
        # clone for actor, critic networks
        self.optimizer.actor_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        self.optimizer.target_actor_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        self.optimizer.critic_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        self.optimizer.target_critic_keras_optimizer = clone_optimizer(
            self.optimizer.keras_optimizer)
        del self.optimizer.keras_optimizer

        super(DDPG, self).compile(memory, self.optimizer, policy, preprocessor)

    def build_actor_models(self, weight_init):
        model = self.Sequential()
        self.build_hidden_layers(model, normal_02)
        model.add(self.Dense(self.env_spec['action_dim'],
                             init='uniform',
                             # activation=self.output_layer_activation))
                             activation='tanh2'))
        logger.info('Actor model summary')
        model.summary()
        self.actor = model
        self.target_actor = clone_model(self.actor)

    def build_critic_models(self, weight_init):
        state_branch = self.Sequential()
        state_branch.add(self.Dense(
            self.hidden_layers[0] if len(self.hidden_layers) > 1 else math.floor(self.hidden_layers[0]  * 1.25),
            input_shape=(self.env_spec['state_dim'],),
            activation=self.hidden_layers_activation,
            init='normal'))
        state_branch.add(self.Dense(
            self.hidden_layers[1] if len(self.hidden_layers) > 1 else self.hidden_layers[0],
            activation=self.hidden_layers_activation,
            init='normal'))

        # add action branch to second layer of the network
        action_branch = self.Sequential()
        action_branch.add(self.Dense(
            self.hidden_layers[1] if len(self.hidden_layers) > 1 else self.hidden_layers[0],
            input_shape=(self.env_spec['action_dim'],),
            activation=self.hidden_layers_activation,
            init='normal'))

        input_layer = self.Merge([state_branch, action_branch], mode='concat')

        model = self.Sequential()
        model.add(input_layer)

        # if (len(self.hidden_layers) > 1):
        #     for i in range(2, len(self.hidden_layers)):
        #         model.add(self.Dense(
        #             self.hidden_layers[i],
        #             init=normal_02,
        #             # use_bias=True,
        #             activation=self.hidden_layers_activation))

        model.add(self.Dense(1,
                             init='uniform',
                             activation=self.output_layer_activation))
        logger.info('Critic model summary')
        model.summary()
        self.critic = model
        self.target_critic = clone_model(self.critic)

    def build_model(self):
        self.build_actor_models(self.weight_init)
        self.build_critic_models(self.weight_init)

    def custom_critic_loss(self, y_true, y_pred):
        return self.K.mean(self.K.square(y_true - y_pred))

    def compile_model(self):
        self.actor_state = self.actor.inputs[0]
        self.action_gradient = self.K.placeholder(
            shape=(None, self.env_spec['action_dim']))
        self.actor_grads = self.K.tf.gradients(
            self.actor.output, self.actor.trainable_weights,
            -self.action_gradient)
        self.actor_optimize = self.K.tf.train.AdamOptimizer(
            self.lr_actor).apply_gradients(
            zip(self.actor_grads, self.actor.trainable_weights))

        self.critic_state = self.critic.inputs[0]
        self.critic_action = self.critic.inputs[1]
        self.critic_action_grads = self.tf.gradients(
            self.critic.output, self.critic_action)

        self.target_actor.compile(
            loss='mse',
            optimizer=self.optimizer.target_actor_keras_optimizer)
        logger.info("Actor Models compiled")

        self.critic.compile(
            loss=self.custom_critic_loss,
            optimizer=self.optimizer.critic_keras_optimizer)
        self.target_critic.compile(
            loss='mse',
            optimizer=self.optimizer.target_critic_keras_optimizer)
        logger.info("Critic Models compiled")

        init_op = self.tf.global_variables_initializer()
        self.sess.run(init_op)
        logger.info("Tensorflow variables initializaed")

    def update(self, sys_vars):
        '''Agent update apart from training the Q function'''
        self.policy.update(sys_vars)
        self.update_n_epoch(sys_vars)

    def train_critic(self, minibatch):
        '''update critic network using K-mean loss'''
        mu_prime = self.target_actor.predict(minibatch['next_states'])
        Q_prime = self.target_critic.predict(
            [minibatch['next_states'], mu_prime])
        y = minibatch['rewards'] + self.gamma * \
            (1 - minibatch['terminals']) * Q_prime
        critic_loss = self.critic.train_on_batch(
            [minibatch['states'], minibatch['actions']], y)

        return critic_loss

    def train_actor(self, minibatch):
        '''update actor network using sampled gradient'''
        actions = self.actor.predict(minibatch['states'])
        critic_grads = self.sess.run(
            self.critic_action_grads, feed_dict={
                self.critic_state: minibatch['states'],
                self.critic_action: actions
            })[0]
        # actor.train(minibatch['states'], critic_grads)
        self.sess.run(self.actor_optimize, feed_dict={
            self.actor_state: minibatch['states'],
            self.action_gradient: critic_grads
        })

        actor_loss = 0
        return actor_loss

    def train_target_networks(self):
        '''update both target networks'''
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i, _w in enumerate(actor_weights):
            target_actor_weights[i] = self.TAU * actor_weights[i] + (
                1 - self.TAU) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i, _w in enumerate(critic_weights):
            target_critic_weights[i] = self.TAU * critic_weights[i] + (
                1 - self.TAU) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

    def train_an_epoch(self):
        minibatch = self.memory.rand_minibatch(self.batch_size)
        critic_loss = self.train_critic(minibatch)
        actor_loss = self.train_actor(minibatch)
        self.train_target_networks()

        loss = critic_loss + actor_loss
        return loss
