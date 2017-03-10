import numpy as np
from rl.memory.linear import LinearMemory
from rl.util import log_self
import math


class HighLowMemory(LinearMemory):

    '''
    Memory divided into two: good and bad experiences
    As with RankedMemory experiences are grouped episodically
    Episodes with a total reward > threshold are assigned to good memory
    The threshold is recomputed every n episodes and
    episodes are reassigned accordingly.
    Memories are sampled from good experiences with a self.prob_high
    Memories are sampled from bad experiences with a 1 - self.prob_high
    Experiences are sampled from a maximum of 3 randomly selected episodes,
    per minibatch for each of the high and low memories
    TODO improvement: do a more natural continuous range to sort high low
    by self.epi_memory.sort(key=lambda epi_exp: epi_exp['total_rewards'])
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking
        super(HighLowMemory, self).__init__()
        # use the old self.exp as buffer, remember to clear
        self.last_exp = self.exp
        self.epi_memory_high = []
        self.epi_memory_low = []
        self.max_reward = -math.inf
        self.min_reward = math.inf
        # 1st  5 epis goes into bad half, recompute every 5 epis
        self.threshold = math.inf
        self.threshold_history = []
        self.epi_num = 0
        self.prob_high = 0.66
        self.num_epis_to_sample = 3
        self.max_epis_in_mem = 15
        self.recompute_freq = 10
        log_self(self)

    def reassign_episodes(self):
        new_high, new_low = []

        for mem in (self.epi_memory_high, self.epi_memory_low):
            for epi_exp in mem:
                if (epi_exp['total_rewards'] > self.threshold):
                    new_high.append(epi_exp)
                else:
                    new_low.append(epi_exp)

        self.epi_memory_high = new_high
        self.epi_memory_low = new_low

    def compute_threshold(self):
        self.threshold_history.append([self.threshold,
                                       self.max_reward,
                                       self.min_reward])
        if (len(self.threshold_history) > 1):
            # Scaled because this threshold seems too severe based on trial
            # runs
            self.threshold =  \
                max(self.threshold,
                    (self.max_reward + self.min_reward) / 2.0 * 0.75)
        else:
            self.threshold = (self.max_reward + self.min_reward) / 2.0 * 0.75
        self.reassign_episodes()
        self.max_reward = -math.inf
        self.min_reward = math.inf

    def add_exp(self, action, reward, next_state, terminal):
        super(HighLowMemory, self).add_exp(
            action, reward, next_state, terminal)
        if terminal:
            epi_exp = {
                'exp': self.exp,
                'total_rewards': np.sum(self.exp['rewards']),
                'epi_num': self.epi_num
            }
            if (epi_exp['total_rewards'] <= self.threshold):
                self.epi_memory_low.append(epi_exp)
            else:
                self.epi_memory_high.append(epi_exp)
            if (self.epi_num > 0 and self.epi_num % self.recompute_freq == 0):
                self.compute_threshold()
            if (epi_exp['total_rewards'] > self.max_reward):
                self.max_reward = epi_exp['total_rewards']
            if (epi_exp['total_rewards'] < self.min_reward):
                self.min_reward = epi_exp['total_rewards']
            self.last_exp = self.exp
            self.exp = {k: [] for k in self.exp_keys}
            self.epi_num += 1
            # print("THRESHOLD HISTORY")
            # print(self.threshold_history)
            # print("HIGH MEM")
            # for epi in self.epi_memory_high:
            #     print(str(epi['total_rewards'])+ " ,", end=" ")
            # print()
            # print("LOW MEM")
            # for epi in self.epi_memory_low:
            #     print(str(epi['total_rewards'] )+ " ,", end=" ")
            # print()

    def pop(self):
        '''convenient method to get exp at [last_ind]'''
        buffer_exp = self.exp  # store for restore later
        self.exp = self.last_exp
        res = super(HighLowMemory, self).pop()
        self.exp = buffer_exp
        return res

    def rand_minibatch(self, size):
        # base case, early exit
        high_samples = np.int(np.ceil(size * self.prob_high))
        low_samples = size - high_samples

        if (len(self.epi_memory_high) == 0 and
                len(self.epi_memory_low) == 0):
            return super(HighLowMemory, self).rand_minibatch(size)

        if (len(self.epi_memory_high) == 0):
            high_samples = 0
            low_samples = size

        high_samples_per_epi = np.int(
            np.ceil(high_samples / self.num_epis_to_sample))
        low_samples_per_epi = np.int(
            np.ceil(low_samples / self.num_epis_to_sample))

        buffer_exp = self.exp
        minibatch_as_list = []
        if high_samples > 0:
            for _i in range(4):
                idx = np.random.randint(0, len(self.epi_memory_high))
                epi_exp = self.epi_memory_high[idx]['exp']
                self.exp = epi_exp
                epi_minibatch = super(HighLowMemory, self).rand_minibatch(
                    high_samples_per_epi)
                minibatch_as_list.append(epi_minibatch)

        if low_samples > 0:
            for _i in range(4):
                idx = np.random.randint(0, len(self.epi_memory_low))
                epi_exp = self.epi_memory_low[idx]['exp']
                self.exp = epi_exp
                epi_minibatch = super(HighLowMemory, self).rand_minibatch(
                    low_samples_per_epi)
                minibatch_as_list.append(epi_minibatch)

        # set buffer back to original
        self.exp = buffer_exp

        # merge all minibatches from best_epi_memory into a minibatch
        minibatch = {}
        for k in self.exp_keys:
            k_exp = np.concatenate(
                [epi_exp[k] for epi_exp in minibatch_as_list]
            )[-size:]
            minibatch[k] = k_exp
        assert len(minibatch['rewards']) == size, 'minibatch has the wrong size'

        return minibatch


class HighLowMemoryWithForgetting(HighLowMemory):

    '''
    Like HighLowMemory but also has forgetting capability
    Controlled by max_epis_in_mem param
    '''

    def __init__(self, **kwargs):  # absorb generic param without breaking
        super(HighLowMemoryWithForgetting, self).__init__()
        self.max_epis_in_mem = 250
        log_self(self)

    def reassign_episodes(self):
        new_high, new_low = []

        for mem in (self.epi_memory_high, self.epi_memory_low):
            for epi_exp in mem:
                if (self.epi_num - epi_exp['epi_num'] <= self.max_epis_in_mem):
                    if (epi_exp['total_rewards'] > self.threshold):
                        new_high.append(epi_exp)
                    else:
                        new_low.append(epi_exp)

        self.epi_memory_high = new_high
        self.epi_memory_low = new_low
