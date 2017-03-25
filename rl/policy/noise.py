import numpy as np
from rl.util import logger
from rl.policy.base_policy import Policy


class AnnealedGaussian(Policy):

    '''
    Noise policy, mainly for DDPG.
    Original inspiration from
    https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py
    '''

    def __init__(self, env_spec,
                 mu, sigma, sigma_min,
                 init_e=1.0, final_e=0.1, exploration_anneal_episodes=30,
                 **kwargs):  # absorb generic param without breaking
        super(AnnealedGaussian, self).__init__(env_spec)
        self.size = self.env_spec['action_dim']
        self.n_steps_annealing = self.env_spec['timestep_limit'] / 2
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0
        self.init_e = init_e
        self.final_e = final_e
        self.e = self.init_e
        self.exploration_anneal_episodes = exploration_anneal_episodes

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(self.n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

    def select_action(self, state):
        agent = self.agent
        state = np.expand_dims(state, axis=0)
        if self.env_spec['actions'] == 'continuous':
            action = agent.actor.predict(state)[0] + self.sample()
        else:
            if self.e > np.random.rand():
                action = np.random.choice(agent.env_spec['actions'])
            else:
                Q_state = agent.actor.predict(state)[0]
                assert Q_state.ndim == 1
                action = np.argmax(Q_state)
                # logger.info(str(Q_state)+' '+str(action))
        return action

    def update(self, sys_vars):
        '''strategy to update epsilon in agent'''
        epi = sys_vars['epi']
        rise = self.final_e - self.init_e
        slope = rise / float(self.exploration_anneal_episodes)
        self.e = max(slope * epi + self.init_e, self.final_e)
        return self.e


class GaussianWhiteNoise(AnnealedGaussian):

    def __init__(self, env_spec,
                 mu=0., sigma=.3, sigma_min=None,
                 **kwargs):  # absorb generic param without breaking
        super(GaussianWhiteNoise, self).__init__(
            env_spec, mu=mu, sigma=sigma, sigma_min=sigma_min)

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample


class OUNoise(AnnealedGaussian):

    '''
    Based on
    http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    '''

    def __init__(self, env_spec,
                 theta=.15, mu=0., sigma=.3, dt=1e-2, x0=None, sigma_min=None,
                 **kwargs):  # absorb generic param without breaking
        super(OUNoise, self).__init__(
            env_spec, mu=mu, sigma=sigma, sigma_min=sigma_min,
            **kwargs)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset_states()

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    def sample(self):
        x = self.x_prev + self.theta * \
            (self.mu - self.x_prev) * self.dt + self.current_sigma * \
            np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x
