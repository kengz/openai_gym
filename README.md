# OpenAI Gym [![CircleCI](https://circleci.com/gh/kengz/openai_gym.svg?style=shield)](https://circleci.com/gh/kengz/openai_gym) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a0e6bbbb6c4845ccaab2db9aecfecbb0)](https://www.codacy.com/app/kengzwl/openai_gym?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_gym&amp;utm_campaign=Badge_Grade)

[OpenAI Gym Doc](https://gym.openai.com/docs) | [OpenAI Gym Github](https://github.com/openai/gym) | [RL intro](https://gym.openai.com/docs/rl)

Working out at the (OpenAI) gym.

This was started from the Eligible Reinforcement Learning event, and we just kept working on it. Checkout the tutorial session videos here: [Part 1](https://youtu.be/qBhLoeijgtA), [Part 2](https://youtu.be/wNSlZJGdodE)


## Installation

### Basic

```shell
git clone https://github.com/kengz/openai_gym.git
cd openai_gym
python setup.py install
```

*Note that by default it installs Tensorflow for Python3 on MacOS. [If you're on a different platform, choose the correct binary to install from TF.](https://www.tensorflow.org/get_started/os_setup#pip_installation)*

```shell
# default option
TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
# install from the TF_BINARY_URL
sudo pip install --upgrade $TF_BINARY_URL
```

### Complete

To run more than just the classic control gym env, we need to install the OpenAI gym fully. We refer to the [Install Everything](https://github.com/openai/gym#installing-everything) of the repo (which is still broken at the time of writing).

```shell
brew install cmake boost boost-python sdl2 swig wget
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[all]'
```

Try to run a Lunar Lander env, it will break (unless they fix it):
```python
import gym
env = gym.make('LunarLander-v2')
env.reset()
env.render()
```

If it fails, debug as follow (and repeat once more if it fails again, glorious python):

```shell
pip3 uninstall Box2D box2d-py
git clone https://github.com/pybox2d/pybox2d
cd pybox2d/
python setup.py clean
python setup.py build
python setup.py install
```

To run Atari envs three additional dependencies are required

```shell
pip install atari_py
pip install Pillow
pip install PyOpenGL
```

Then check that it works with
```python
import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()
```

## Usage

### Data files auto-sync (optional)

For auto-syncing `data/` files, we use Gulp. This sets up a watcher for automatically copying data files via Keybase. If you're not Keng or Laura, change the Keybase filepath in `gulpfile.js`.

```shell
npm install --global gulp-cli
npm install --save-dev gulp gulp-watch gulp-changed
run the
gulp
```

### Run experiments locally

Configure the `"start"` scripts in `package.json` for easily running the same experiments over and over again.

```shell
# easy run command
npm start
# to clear data/
npm run clear
```

To customize your run commands, use plain python:

```shell
python3 main.py -s lunar_dqn -b -g | tee -a ./data/terminal.log
```

The extra flags are:

- `-d`: log debug info. Default: `False`
- `-b`: blind mode, do not render graphics. Default: `False`
- `-s <sess_name>`: specify which of `rl/asset/sess_spec.json` to run. Default: `-s dev_dqn`
- `-t <times>`: the number of sessions to run per experiment. Default: `1`
- `-p`: run param selection. Default: `False`
- `-l`: use line search for param selection. Default: `False`
- `-g`: plot graphs live. Default: `False`


### Run experiments remotely

Log in via ssh, start a screen, run, then detach screen.

```shell
screen
# enter the screen
npm run remote
# or full python command goes like
xvfb-run -a -s "-screen 0 1400x900x24" -- python3 main.py -b | tee -a ./data/terminal.log
# use Cmd+A+D to detach from screen, then Cmd+D to disconnect ssh
# use screen -r to resume screen next time
```


## Development

This is still under active development, and documentation is sparse. The main code lives inside `rl/`.

The design of the code is clean enough to simply infer how things work by example.

- `data/`: contains all the graphs per experiment sessions, JSON data file per experiment, and csv metrics dataframe per run of multiple experiments
- `rl/agent/`: custom agents. Refer to `base_agent.py` and `dqn.py` to build your own
- `rl/asset/`: specify new problems and sess_specs to run experiments for.
- `rl/model/`: if you decide to save a model, this is the place
- `rl/experiment.py`: the main high level experiment logic.
- `rl/memory.py`: RL agent memory classes
- `rl/policy.py`: RL agent policy classes
- `rl/policy.py`: RL agent preprocessor (state and memory) classes
- `rl/util.py`: Generic util

Each run is by specifying a `sess_name` or `sess_id`. This runs experiments sharing the same `prefix_id`. Each experiment runs multiple sessions to take the average metrics and plot graphs. At last the experiments are aggregated into a metrics dataframe, sorted by the best experiments. All these data and graphs are saved into a new folder in `data/` named with the `prefix_id`.


## Roadmap

Check the latest under the [Github Projects](https://github.com/kengz/openai_gym/projects)

## Authors

- Wah Loon Keng
- Laura Graesser
