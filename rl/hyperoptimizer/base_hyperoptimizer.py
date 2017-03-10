import copy
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from rl.util import logger, timestamp, PARALLEL_PROCESS_NUM, debug_mem_usage


class HyperOptimizer(object):

    '''
    The base class of hyperparam optimizer, with core methods
    read about it on the documentation
    input: Trial (and some specs), param space P (as standardized specs)
    Algo:
    1. search the next p in P using its internal search algo,
    add to its internal `param_search_list`
    2. run a (slow) function Trial(p) = score (inside trial data)
    3. update search using feedback score
    4. repeat till max steps or fitness condition met

    it will be ran by the experiment as:
    hyperopt = HyperOptimizer(Trial, **experiment_kwargs)
    experiment_data = hyperopt.run()
    '''

    def __init__(self, Trial, **kwargs):
        self.Trial = Trial
        self.REQUIRED_ARGS = [
            'experiment_spec',
            'experiment_id_override',
            'times'
        ]
        self.PARALLEL_PROCESS_NUM = PARALLEL_PROCESS_NUM
        self.free_cpu = self.PARALLEL_PROCESS_NUM  # for parallel run
        logger.info('Initialize {}'.format(self.__class__.__name__))
        self.set_keys(**kwargs)
        self.init_search()

    def set_keys(self, **kwargs):
        assert all(k in kwargs for k in self.REQUIRED_ARGS), \
            'kwargs do not have all REQUIRED_ARGS'
        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.experiment_name = self.experiment_spec.get('experiment_name')
        self.run_timestamp = timestamp()
        self.experiment_id = self.experiment_id_override or '{}-{}'.format(
            self.experiment_name, self.run_timestamp)
        self.experiment_data = []
        self.param_search_list = []
        # the index of next param to try in param_search_list
        self.next_trial_num = len(self.param_search_list)

        self.default_param = self.experiment_spec['param']
        unordered_param_range = self.experiment_spec['param_range']
        # import ordering for param_range for search serialization
        self.param_range = OrderedDict(sorted(unordered_param_range.items()))
        self.param_range_keys = sorted(self.param_range.keys())

    def compose_experiment_spec(self, param):
        new_experiment_spec = copy.deepcopy(self.experiment_spec)
        new_experiment_spec.pop('param_range', None)
        new_experiment_spec.update({
            'param': param,
        })
        return new_experiment_spec

    def init_search(self):
        '''initialize the search algo and the search space'''
        raise NotImplementedError()

    def search(self):
        '''
        algo step 1, search and return the next p for Trial(p),
        Its only job is to append to (or modify)
        its internal self.param_search_list using its search logic
        It may refer to self.experiment_data as search memory
        and whatever new pointer or special memory implemented by a HyperOptimizer class
        '''
        raise NotImplementedError()

    def next_param(self):
        '''retrieve trial_num and param, advance the class next_trial_num'''
        assert self.next_trial_num < len(self.param_search_list), \
            'param_search_list expansion cannot keep up with next_trial_num'
        trial_num = self.next_trial_num
        param = self.param_search_list[self.next_trial_num]
        self.next_trial_num = self.next_trial_num + 1
        return (trial_num, param)

    def run_trial(self, trial_num, param):
        '''
        algo step 2, construct and run Trial with the next param
        args trial_num, param must be provided externally,
        otherwise they will not progress within mp.process
        '''
        experiment_spec = self.compose_experiment_spec(param)
        trial = self.Trial(
            experiment_spec, trial_num=trial_num,
            times=self.times,
            num_of_trials=self.num_of_trials,
            run_timestamp=self.run_timestamp,
            experiment_id_override=self.experiment_id_override)
        trial_data = trial.run()
        del trial
        import gc
        gc.collect()
        debug_mem_usage()
        return trial_data

    # retrieve the trial_num, param, fitness_score from trial_data
    @classmethod
    def get_fitness(cls, trial_data):
        trial_id = trial_data['trial_id']
        trial_num = trial_id.split('_').pop()
        param = trial_data['experiment_spec']['param']
        metrics = trial_data['metrics']
        fitness_score = metrics['fitness_score']
        return trial_num, param, fitness_score

    def update_search(self):
        '''algo step 3, update search algo using self.experiment_data'''
        raise NotImplementedError()

    def to_terminate(self):
        '''algo step 4, terminate when at max steps or fitness condition met'''
        raise NotImplementedError()

    # handler task after a search is complete from multiprocessing pool
    def post_search(self, trial_data):
        self.experiment_data.append(trial_data)
        self.update_search()
        self.free_cpu += 1

    @classmethod
    def pool_init(self):
        # you can never be too safe in multiprocessing gc
        import gc
        gc.collect()

    @classmethod
    def raise_error(cls, e):
        logger.error('Pool worker throws Exception')
        print(e.__cause__)
        time.sleep(1)
        os._exit(1)

    def run(self):
        '''
        top level method to run the entire hyperoptimizer
        will gather and compose experiment_data, then return it
        '''
        logger.info('Run {}'.format(self.__class__.__name__))
        # crucial maxtasksperchild to free up memory by respawning worker
        pool = mp.Pool(self.PARALLEL_PROCESS_NUM,
                       initializer=self.pool_init, maxtasksperchild=1)
        while (not self.to_terminate()):
            if self.free_cpu > 0:
                self.free_cpu -= 1  # update
                self.search()  # add to self.param_search_list
                trial_num, param = self.next_param()
                pool.apply_async(
                    self.run_trial, (trial_num, param),
                    callback=self.post_search, error_callback=self.raise_error)
            else:
                pass  # keep looping till free_cpu available
            time.sleep(0.02)  # prevent cpu overwork from while loop
        pool.close()
        pool.join()
        return self.experiment_data
