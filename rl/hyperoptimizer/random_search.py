import json
import numpy as np
from rl.analytics import ideal_fitness_score
from rl.hyperoptimizer.base_hyperoptimizer import HyperOptimizer
from rl.util import PROBLEMS, to_json, logger


class RandomSearch(HyperOptimizer):

    '''
    Random Search by sampling on hysphere around a search path
    algo:
    1. init x a random position in space
    2. until termination (max_eval or fitness, e.g. solved all), do:
        2.1 sample new pos some radius away: next_x = x + r
        2.2 if f(next_x) > f(x) then set x = next_x

    Extra search memory units:
    - search_path
    - best_point

    save for experiment resume, search_history:
    - search_path
    - best_point
    - param_search_list
    '''

    # # calculate the constant radius needed to traverse unit cube
    # def cube_traversal_radius(self):
    #     traversal_diameter = 1/np.power(self.max_evals,
    #                                     1/self.search_dim)
    #     traversal_radius = traversal_diameter/2
    #     return traversal_radius

    def decay_search_radius(self):
        '''
        start of half cube for diameter (0.25 radius) then decay
        at 100 searches, will shrink to 1/10 of initial radius 0.025
        clip to prevent going too small (0.01)
        '''
        min_radius = 0.01
        linear_decay_rate = self.next_trial_num/10./self.PARALLEL_PROCESS_NUM
        self.search_radius = np.clip(
            self.init_search_radius / linear_decay_rate,
            min_radius, self.init_search_radius)

    @classmethod
    def sample_hypersphere(cls, dim, r=1):
        '''Marsaglia algo for sampling uniformly on a hypersphere'''
        v = np.random.randn(dim)
        v = v * r / np.linalg.norm(v)
        return v

    def sample_cube(self):
        return np.random.rand(self.search_dim)

    def sample_r(self):
        return self.sample_hypersphere(
            self.search_dim, self.search_radius)

    # biject [0, 1] to [x_min, x_max]
    def biject_continuous(self, norm_val, x_min, x_max):
        return np.around(norm_val*(x_max - x_min) + x_min, self.precision)

    # biject [0, 1] to x_list = [a, b, c, ...] by binning
    def biject_discrete(self, norm_val, x_list):
        list_len = len(x_list)
        inds = np.arange(list_len)
        cont_val = self.biject_continuous(norm_val, 0, list_len)
        ind = np.digitize(cont_val, inds) - 1
        return x_list[ind]

    # biject one dimension: [0, 1] to a param_range val
    def biject_dim(self, norm_val, dim_spec):
        if isinstance(dim_spec, list):  # discrete
            return self.biject_discrete(norm_val, dim_spec)
        else:  # cont
            return self.biject_continuous(
                norm_val, dim_spec['min'], dim_spec['max'])
        return

    # biject a vector on unit cube into a param in param_space
    def biject_param(self, v):
        param = {}
        for i, param_key in enumerate(self.param_range_keys):
            dim_spec = self.param_range[param_key]
            param[param_key] = self.biject_dim(v[i], dim_spec)
        return param

    def init_search(self):
        '''
        Initialize the random search internal variables
        '''
        self.max_evals = self.experiment_spec['param']['max_evals']
        self.num_of_trials = self.max_evals
        self.search_dim = len(self.param_range_keys)
        self.precision = 4  # decimal roundoff biject_continuous
        self.search_radius = self.init_search_radius = 0.5
        self.search_path = []
        self.best_point = {
            'trial_num': None,
            'param': None,
            'x': self.sample_cube(),
            'fitness_score': float('-inf'),
        }
        problem = PROBLEMS.get(self.experiment_spec['problem'])
        solved_mean_reward = problem['SOLVED_MEAN_REWARD']
        max_episodes = problem['MAX_EPISODES']
        solved_epi_speedup = 3 if solved_mean_reward > 0 else 1./3
        self.ideal_fitness_score = ideal_fitness_score(
            solved_mean_reward, max_episodes, solved_epi_speedup)
        logger.info(
            'ideal_fitness_scrore: {}'.format(self.ideal_fitness_score))

        self.filename = './data/{}/random_search_history.json'.format(
            self.experiment_id)
        if self.experiment_id_override is not None:
            self.load()  # resume

    def search(self):
        '''
        algo step 2.1 sample new pos some radius away: next_x = x + r
        update search_path and param_search_list
        '''
        if self.next_trial_num < len(self.search_path):  # resuming
            next_x = self.search_path[self.next_trial_num]
            next_param = self.param_search_list[self.next_trial_num]
        else:
            next_x = np.clip(self.best_point['x'] + self.sample_r(), 0., 1.)
            next_param = self.biject_param(next_x)
            self.search_path.append(next_x)
            self.param_search_list.append(next_param)

    def update_search(self):
        '''
        algo step 2.2 if f(next_x) > f(x) then set x = next_x
        invoked right after the latest run_trial()
        update self.best_point
        '''
        if (self.next_trial_num < self.PARALLEL_PROCESS_NUM or
                self.next_trial_num < len(self.search_path)):
            # yet to have history or still resuming from history
            return
        assert len(self.experiment_data) > 0, \
            'self.experiment_data must not be empty for update_search'

        self.decay_search_radius()

        x = self.search_path[-1]
        trial_data = self.experiment_data[-1]
        trial_num, param, fitness_score = self.get_fitness(trial_data)
        if fitness_score > self.best_point['fitness_score']:
            self.best_point = {
                'trial_num': trial_num,
                'param': param,
                'x': x,
                'fitness_score': fitness_score,
            }
        self.save()

    def save(self):
        search_history = {
            'search_path': self.search_path,
            'best_point': self.best_point,
            'param_search_list': self.param_search_list,
        }
        with open(self.filename, 'w') as f:
            f.write(to_json(search_history))
        logger.info(
            'Save search history to {}'.format(self.filename))
        return

    def load(self):
        try:
            search_history = json.loads(open(self.filename).read())
            self.search_path = search_history['search_path']
            self.best_point = search_history['best_point']
            self.param_search_list = search_history['param_search_list']
            logger.info('Load search history from {}'.format(self.filename))
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(
                'Fail to load search history from {}'.format(self.filename))
            return None

    def satisfy_fitness(self):
        '''
        break on the first strong solution
        '''
        best_fitness_score = self.best_point['fitness_score']
        if self.next_trial_num < self.PARALLEL_PROCESS_NUM:
            return False
        elif best_fitness_score > self.ideal_fitness_score:
            logger.info(
                'fitness_score {} > ideal_fitness_score {}, terminate'.format(
                    best_fitness_score, self.ideal_fitness_score))
            return True
        else:
            return False

    def to_terminate(self):
        return (self.next_trial_num >= self.max_evals or
                self.satisfy_fitness())
