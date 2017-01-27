import unittest
import pandas as pd
from os import environ
environ['CI'] = environ.get('CI') or 'true'
from rl.experiment import run


class DQNTest(unittest.TestCase):

    def test_gym_tour(self):
        metrics_df = run('dummy')
        assert isinstance(metrics_df, pd.DataFrame)

    def test_q_table(self):
        metrics_df = run('q_table')
        assert isinstance(metrics_df, pd.DataFrame)

    def test_dqn(self):
        metrics_df = run('dqn')
        assert isinstance(metrics_df, pd.DataFrame)

    def test_double_dqn(self):
        metrics_df = run('double_dqn')
        assert isinstance(metrics_df, pd.DataFrame)

    def test_dev_dqn_pass(self):
        metrics_df = run('dev_dqn')
        max_total_rewards = metrics_df['max_total_rewards_stats_max'][0]
        print(max_total_rewards)
        assert max_total_rewards > 100
