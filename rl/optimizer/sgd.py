from rl.optimizer.base_optimizer import Optimizer


class SGDOptimizer(Optimizer):

    '''
    Stochastic gradient descent
    Potential param:
        lr (learning rate)
        momentum
        decay
        nesterov
    '''

    def __init__(self, **kwargs):
        from keras.optimizers import SGD
        self.SGD = SGD

        self.optim_param_keys = ['lr', 'momentum', 'decay', 'nesterov']
        super(SGDOptimizer, self).__init__(**kwargs)

    def init_optimizer(self):
        self.keras_optimizer = self.SGD(**self.optim_param)
