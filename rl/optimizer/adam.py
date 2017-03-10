from rl.optimizer.base_optimizer import Optimizer


class AdamOptimizer(Optimizer):

    '''
    Adam optimizer
    Potential param:
        lr (learning rate)
        beta_1
        beta_2
        epsilon
        decay
        Suggested to leave at default param with the expected of lr
    '''

    def __init__(self, **kwargs):
        from keras.optimizers import Adam
        self.Adam = Adam

        self.optim_param_keys = ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay']
        super(AdamOptimizer, self).__init__(**kwargs)

    def init_optimizer(self):
        self.keras_optimizer = self.Adam(**self.optim_param)
