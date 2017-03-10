import copy
from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer


class LineSearch(HyperOptimizer):

    def init_search(self):
        '''
        convert a dict of param ranges into
        a list parameter settings corresponding
        to a line search of the param range
        for each param
        All other parameters set to default vals
        note that this is order-preserving, as required by design
        '''
        for key in self.param_range_keys:
            vals = self.param_range[key]
            for val in vals:
                param = copy.deepcopy(self.default_param)
                param[key] = val
                self.param_search_list.append(param)
        self.num_of_trials = len(self.param_search_list)

    def search(self):
        '''no action needed here for exhaustive trials'''
        return

    def update_search(self):
        '''no action needed here for exhaustive trials'''
        return

    def to_terminate(self):
        return not (self.next_trial_num < len(self.param_search_list))
