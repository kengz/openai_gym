import copy
import itertools
from rl.hyperoptimizer.line_search import LineSearch


class GridSearch(LineSearch):

    def init_search(self):
        '''
        convert a dict of param ranges into
        a list of cartesian products of param_range
        e.g. {'a': [1,2], 'b': [3]} into
        [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
        note that this is order-preserving, as required by design
        '''
        range_vals = self.param_range.values()
        for vals in itertools.product(*range_vals):
            param = copy.deepcopy(self.default_param)
            param.update(dict(zip(self.param_range_keys, vals)))
            self.param_search_list.append(param)
        self.num_of_trials = len(self.param_search_list)
