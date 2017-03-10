import unittest
import pytest
from os import environ
from rl.experiment import run
from . import conftest
import pandas as pd


class DQNTest(unittest.TestCase):

    @classmethod
    def test_gym_tour(cls):
        print()
        data_df = run('dummy')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_q_table(cls):
        data_df = run('q_table')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dqn(cls):
        data_df = run('dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_rand_dqn(cls):
        data_df = run('rand_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_double_dqn(cls):
        data_df = run('double_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_sarsa(cls):
        data_df = run('sarsa')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_exp_sarsa(cls):
        data_df = run('exp_sarsa')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_offpol_sarsa(cls):
        data_df = run('offpol_sarsa')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_acrobot(cls):
        data_df = run('acrobot')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_mountain_dqn(cls):
        data_df = run('mountain_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @unittest.skipIf(environ.get('CI'),
                     "OpenAI Gym Box2D swigconstant error, issue #100")
    @classmethod
    def test_lunar_dqn(cls):
        data_df = run('lunar_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_breakout_dqn(cls):
        data_df = run('breakout_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_breakout_double_dqn(cls):
        data_df = run('breakout_double_dqn')
        assert isinstance(data_df, pd.DataFrame)

    @classmethod
    def test_dev_dqn_pass(cls):
        data_df = run('dqn')
        max_total_rewards = data_df['max_total_rewards_stats_mean'][0]
        print(max_total_rewards)
        assert max_total_rewards > 50
