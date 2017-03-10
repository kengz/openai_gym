from rl.util import log_self, logger


class Optimizer(object):

    '''
    The base class of Optimizer, with the core methods
    '''

    def __init__(self, **kwargs):
        '''Construct externally, and set at Agent.compile()'''
        self.agent = None
        self.keras_optimizer = None
        self.optim_param = {}
        self.update_optim_param(**kwargs)
        self.init_optimizer()
        log_self(self)

    def update_optim_param(self, **kwargs):
        o_param = {
            k: kwargs.get(k) for k in self.optim_param_keys
            if kwargs.get(k) is not None}
        self.optim_param.update(o_param)

    def init_optimizer(self):
        raise NotImplementedError()

    def change_optim_param(self, **new_param):
        self.update_optim_param(**new_param)
        self.init_optimizer()
        logger.info("Optimizer param changed")
        log_self(self)
