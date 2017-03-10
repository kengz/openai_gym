import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


# explicitly config
test_args = [
    '--cov-report=term',
    '--cov-report=html',
    '--cov=rl',
    'test'
]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# the setup
setup(
    name='openai_lab',
    version='1.0.0',
    description='An experimentation system for Reinforcement Learning using OpenAI and Keras',
    long_description=read('README.md'),
    keywords='openai gym',
    url='https://github.com/kengz/openai_lab',
    author='kengz,lgraesser',
    author_email='kengzwl@gmail.com',
    license='MIT',
    packages=[],
    zip_safe=False,
    include_package_data=True,
    install_requires=[],
    dependency_links=[],
    extras_require={
        'dev': [],
        'docs': [],
        'testing': []
    },
    classifiers=[],
    tests_require=['pytest', 'pytest-cov'],
    test_suite='test',
    cmdclass={'test': PyTest}
)
