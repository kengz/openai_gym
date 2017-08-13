from rl.optimizer.base_optimizer import Optimizer

from torch import nn
from torch.optim import adam

def both(a, b):
    if a is None or b is None:
        return None
    else:
        return (a, b)

def filter_non_none_values(d):
    return { k : v for k, v in d.items() if v is not None }


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
        self.optim_param_keys = ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay']
        super(AdamOptimizer, self).__init__(**kwargs)

    def init_optimizer(self):
        from keras.optimizers import Adam
        self.keras_optimizer = Adam(**self.optim_param)

    def torch_optimizer(self, parameters: nn.ParameterList):
        kwargs = filter_non_none_values({
            "lr": self.optim_param.get("lr"),
            "betas": both(self.optim_param.get("beta_1"), self.optim_param.get("beta_2")),
            "eps": self.optim_param.get("epsilon"),
            "weight_decay": self.optim_param.get("decay")
        })
        return adam.Adam(parameters, **kwargs)

