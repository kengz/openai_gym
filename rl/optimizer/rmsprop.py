from rl.optimizer.base_optimizer import Optimizer


class RMSpropOptimizer(Optimizer):

    '''
    RMS prop
    Potential param:
        lr (learning rate)
        rho
        decay
        epsilon
    '''

    def __init__(self, **kwargs):
        from keras.optimizers import RMSprop
        self.RMSprop = RMSprop

        self.optim_param_keys = ['lr', 'rho', 'decay', 'epsilon']
        super(RMSpropOptimizer, self).__init__(**kwargs)

    def init_optimizer(self):
        self.keras_optimizer = self.RMSprop(**self.optim_param)
