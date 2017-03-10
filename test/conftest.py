import pytest
import rl
from os import environ

environ['CI'] = environ.get('CI') or 'true'


def pytest_runtest_setup(item):
    for problem in rl.util.PROBLEMS:
        if problem == 'CartPole-v0':
            rl.util.PROBLEMS[problem]['MAX_EPISODES'] = 30
        else:
            rl.util.PROBLEMS[problem]['MAX_EPISODES'] = 3
