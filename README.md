# OpenAI Lab [![CircleCI](https://circleci.com/gh/kengz/openai_lab.svg?style=shield)](https://circleci.com/gh/kengz/openai_lab) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a0e6bbbb6c4845ccaab2db9aecfecbb0)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_lab&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&utm_medium=referral&utm_content=kengz/openai_lab&utm_campaign=Badge_Coverage) [![GitHub forks](https://img.shields.io/github/forks/kengz/openai_lab.svg?style=social&label=Fork)](https://github.com/kengz/openai_lab) [![GitHub stars](https://img.shields.io/github/stars/kengz/openai_lab.svg?style=social&label=Star)](https://github.com/kengz/openai_lab)

---

<p align="center"><b><a href="http://kengz.me/openai_lab">OpenAI Lab Documentation</a></b></p>

---

_An experimentation system for Reinforcement Learning using OpenAI and Keras._

This lab is created to let us do Reinforcement Learning (RL) like science - _theorize, experiment_. We can theorize as fast as we think, and experiment as fast as the computers can run.

With _OpenAI Lab_, we had solved a few OpenAI environments by running dozens of experiments, each with hundreds of trials. Each new experiment takes minimal effort to setup and run - we will show by example below.

Before this, experiments used to be hard and slow, because we often have to write most things from scratch and reinvent the wheels. To solve this, the lab provides a standard, extensible platform with a host of reusable components.

_OpenAI Lab_ lowers the experimental complexity and enables an explosion of experiments - we can quickly add new RL component, make new combinations, run hyperparameter selection and solve the environments. This unlocks a new perspective to treat RL as a full-on experimental science.

<img alt="Timelapse of OpenAI Lab" src="http://kengz.me/openai_lab/images/lab_demo_dqn.gif" />
_Timelapse of OpenAI Lab (open the gif in new tab for larger size)._


## Lab Demo

Each experiment involves:
- a problem - an [OpenAI Gym environment](https://gym.openai.com/envs)
- a RL agent with modular components `agent, memory, optimizer, policy, preprocessor`, each of which is an experimental variable.

<<<<<<< HEAD
```shell
npm install --global gulp-cli
npm install --save-dev gulp gulp-watch gulp-changed
# run the file watcher
gulp
```
=======
We specify input parameters for the experimental variable, run the experiment, record and analyze the data, conclude if the agent solves the problem with high rewards.
>>>>>>> master

### Specify Experiment

The example below is fully specified in `rl/asset/experiment_specs.json` under `dqn`:

```json
{
  "dqn": {
    "problem": "CartPole-v0",
    "Agent": "DQN",
    "HyperOptimizer": "GridSearch",
    "Memory": "LinearMemoryWithForgetting",
    "Optimizer": "AdamOptimizer",
    "Policy": "BoltzmannPolicy",
    "PreProcessor": "NoPreProcessor",
    "param": {
      "train_per_n_new_exp": 1,
      "lr": 0.001,
      "gamma": 0.96,
      "hidden_layers_shape": [16],
      "hidden_layers_activation": "sigmoid",
      "exploration_anneal_episodes": 20
    },
    "param_range": {
      "lr": [0.001, 0.01, 0.02, 0.05],
      "gamma": [0.95, 0.96, 0.97, 0.99],
      "hidden_layers_shape": [
        [8],
        [16],
        [32]
      ],
      "exploration_anneal_episodes": [10, 20]
    }
  }
}
```

- *experiment*: `dqn`
- *problem*: [CartPole-v0](https://gym.openai.com/envs/CartPole-v0)
- *variable agent component*: `Boltzmann` policy
- *control agent variables*:
    - `DQN` agent
    - `LinearMemoryWithForgetting`
    - `AdamOptimizer`
    - `NoPreProcessor`
- *parameter variables values*: the `"param_range"` JSON

<<<<<<< HEAD
```shell
python3 main.py -bgp -s lunar_dqn -t 5 | tee -a ./data/terminal.log
```
=======
An **experiment** will run a trial for each combination of `param` values; each **trial** will run for multiple repeated **sessions**. For `dqn`, there are `96` param combinations (trials), and `5` repeated sessions per trial. Overall, this experiment will run `96 x 5 = 480` sessions.
>>>>>>> master


### Lab Workflow

The workflow to setup this experiment is as follow:

1. Add the new theorized component `Boltzmann` in `rl/policy/boltzmann.py`
2. Specify `dqn` experiment spec in `experiment_spec.json` to include this new variable,  reuse the other existing RL components, and specify the param range.
3. Add this experiment to the lab queue in `config/production.json`
4. Run `grunt -prod`
5. Analyze the graphs and data (live-synced)

<<<<<<< HEAD
Log in via ssh, start a screen, run, then detach screen.

```shell
screen
# enter the screen
npm run remote
# or full python command goes like
xvfb-run -a -s "-screen 0 1400x900x24" -- python3 main.py -bgp -s lunar_dqn -t 5 | tee -a ./data/terminal.log
# use Cmd+A+D to detach from screen, then Cmd+D to disconnect ssh
# use screen -r to resume screen next time
```
=======
>>>>>>> master

### Lab Results

<img alt="The dqn experiment analytics" src="http://kengz.me/openai_lab/images/dqn.png" />
<img alt="The dqn experiment analytics correlation" src="http://kengz.me/openai_lab/images/dqn_correlation.png" />

_The dqn experiment analytics generated by the lab (open in new tab for larger size). This is a pairplot, where we isolate each variable, flatten the others, plot each trial as a point. The darker the color the higher ratio of the repeated sessions the trial solves._


fitness_score|mean_rewards_per_epi_stats_mean|mean_rewards_stats_mean|epi_stats_mean|solved_ratio_of_sessions|num_of_sessions|max_total_rewards_stats_mean|t_stats_mean|trial_id|variable_exploration_anneal_episodes|variable_gamma|variable_hidden_layers_shape|variable_lr
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
1.178061|1.178061|195.726|169.2|1.0|5|200.0|196.4|dqn-2017_02_27_002407_t44|10|0.99|[32]|0.001
1.173569|1.173569|195.47|168.0|1.0|5|200.0|199.0|dqn-2017_02_27_002407_t32|10|0.97|[32]|0.001
1.152447|1.152447|195.248|170.6|1.0|5|200.0|198.2|dqn-2017_02_27_002407_t64|20|0.96|[16]|0.001
1.128509|1.128509|195.392|177.8|1.0|5|200.0|199.0|dqn-2017_02_27_002407_t92|20|0.99|[32]|0.001
1.127216|1.127216|195.584|175.0|1.0|5|200.0|199.0|dqn-2017_02_27_002407_t76|20|0.97|[16]|0.001

_Analysis data table, top 5 trials._

On completion, from the analytics (zoom on the graph), we conclude that the experiment is a success, and the best agent that solves the problem has the parameters:

- *lr*: 0.001
- *gamma*: 0.99
- *hidden_layers_shape*: [32]
- *exploration_anneal_episodes*: 10


### Run Your Own Lab

Want to run the lab? Go to [Installation](http://kengz.me/openai_lab/installation), [Usage](http://kengz.me/openai_lab/usage) and [Development](http://kengz.me/openai_lab/development).
