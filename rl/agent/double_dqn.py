import numpy as np
import torch

from rl import torch_utils
from rl.agent import dqn
from rl.util import logger, clone_optimizer


# CR-soon fyquah: Is inheritance the best abstraction here? Sure, it saves a
# few lines of code, but the code is quite hard to reason about.
class DoubleDQN(dqn.DQN):

    '''
    The base class of double DQNs
    '''

    def build_model(self):
        super(DoubleDQN, self).build_model()
        self.model_2 = dqn.Net(self)
        torch_util.maybe_cuda(self.model_2.cuda())
        return self.model, self.model_2

    def compile_model(self):
        super().compile_model()
        self.torch_optimizer_2 = self.optimizer.torch_optimizer(
                self.model_2.parameters())
        logger.info("Models 1 and 2 compiled")

    def switch_models(self):
        # Switch model 1 and model 2, also the optimizers
        self.model, self.model_2 = self.model_2, self.model
        self.torch_optimizer, self.torch_optimizer_2 = \
                self.torch_optimizer_2, self.torch_optimizer

    # def recompile_model(self, sys_vars):
    #     '''rotate and recompile both models'''
    #     # TODO fix this, double recompile breaks solving power
    #     if self.epi_change_lr is not None:
    #         self.switch_models()  # to model_2
    #         super(DoubleDQN, self).recompile_model(sys_vars)
    #         self.switch_models()  # back to model
    #         super(DoubleDQN, self).recompile_model(sys_vars)
    #     return self.model

    def compute_Q_states(self, minibatch):
        (Q_states, Q_next_states_select, _max) = \
                super().compute_Q_states(minibatch)
        # Different from (single) dqn: Select max using model 2
        _, Q_next_states_max_ind = torch.max(Q_next_states_select, dim=1)
        # same as dqn again, but use Q_next_states_max_ind above
        Q_next_states = torch.clamp(
                self.model_2(minibatch['next_states']),
                min=-self.clip_val,
                max=self.clip_val)
        rows = torch.arange(
                0, Q_next_states_max_ind.size()[0]).long()
        rows = torch_utils.maybe_cuda(rows)
        # CR-someday fyquah: Unnecessary conversion to-and-fro numpy is a
        # temporary work-around over a bug in pytorch that disallows
        # advanced indexing with GPU tensors in [Q_next_states].
        Q_next_states_max_ind = torch_utils.from_numpy(
                torch_utils.to_numpy(Q_next_states_max_ind.data))
        Q_next_states_max = Q_next_states[rows, Q_next_states_max_ind]
        return (Q_states, Q_next_states, Q_next_states_max)

    def train_an_epoch(self):
        self.switch_models()
        return super(DoubleDQN, self).train_an_epoch()
