# OpenAI Lab [![GitHub release](https://img.shields.io/github/release/kengz/openai_lab.svg)](https://github.com/kengz/openai_lab) [![CircleCI](https://circleci.com/gh/kengz/openai_lab.svg?style=shield)](https://circleci.com/gh/kengz/openai_lab) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kengz/openai_lab&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/9e55f845b10b4b51b213620bfb98e4b3)](https://www.codacy.com/app/kengzwl/openai_lab?utm_source=github.com&utm_medium=referral&utm_content=kengz/openai_lab&utm_campaign=Badge_Coverage) [![GitHub stars](https://img.shields.io/github/stars/kengz/openai_lab.svg?style=social&label=Star)](https://github.com/kengz/openai_lab) [![GitHub forks](https://img.shields.io/github/forks/kengz/openai_lab.svg?style=social&label=Fork)](https://github.com/kengz/openai_lab)

>**MIGRATION NOTICE: We built the next version: [SLM-Lab](https://github.com/kengz/SLM-Lab).**

---

<p align="center"><b><a href="http://kengz.me/openai_lab">OpenAI Lab Documentation</a></b></p>

---

_An experimentation framework for Reinforcement Learning using OpenAI Gym, Tensorflow, and Keras._

_OpenAI Lab_ is created to do Reinforcement Learning (RL) like science - _theorize, experiment_. It provides an easy interface to [OpenAI Gym](https://gym.openai.com/) and [Keras](https://keras.io/), with an automated experimentation and evaluation framework.

### Features

1. **Unified RL environment and agent interface** using OpenAI Gym, Tensorflow, Keras, so you can focus on developing the algorithms.
2. **[Core RL algorithms implementations](http://kengz.me/openai_lab/#agents-matrix), with reusable modular components** for developing deep RL algorithms.
3. **[An experimentation framework](http://kengz.me/openai_lab/#experiments)** for running hundreds of trials of hyperparameter optimizations, with logs, plots and analytics for testing new RL algorithms. Experimental settings are stored in standardized JSONs for reproducibility and comparisons.
4. **[Automated analytics of the experiments](http://kengz.me/openai_lab/#analysis)** for evaluating the RL agents and environments, and to help pick the best solution.
5. **The [Fitness Matrix](http://kengz.me/openai_lab/#fitness-matrix)**, a table of the best scores of RL algorithms v.s. the environments; useful for research.


With OpenAI Lab, we could focus on researching the essential elements of reinforcement learning such as the algorithm, policy, memory, and parameter tuning. It allows us to build agents efficiently using existing components with the implementations from research ideas. We could then test the research hypotheses systematically by running experiments.

*Read more about the research problems the Lab addresses in [Motivations](http://kengz.me/openai_lab/#motivations). Ultimately, the Lab is a generalized framework for doing reinforcement learning, agnostic of OpenAI Gym and Keras. E.g. Pytorch-based implementations are on the roadmap.*


### Implemented Algorithms

A list of the core RL algorithms implemented/planned.

To see their scores against OpenAI gym environments, go to **[Fitness Matrix](http://kengz.me/openai_lab/#fitness-matrix)**.


|algorithm|implementation|eval score (pending)|
|:---|:---|:---|
|[DQN](https://arxiv.org/abs/1312.5602)|[DQN](https://github.com/kengz/openai_lab/blob/master/rl/agent/dqn.py)|-|
|[Double DQN](https://arxiv.org/abs/1509.06461)|[DoubleDQN](https://github.com/kengz/openai_lab/blob/master/rl/agent/double_dqn.py)|-|
|[Dueling DQN](https://arxiv.org/abs/1511.06581)|-|-|
|Sarsa|[DeepSarsa](https://github.com/kengz/openai_lab/blob/master/rl/agent/deep_sarsa.py)|-|
|Off-Policy Sarsa|[OffPolicySarsa](https://github.com/kengz/openai_lab/blob/master/rl/agent/offpol_sarsa.py)|-|
|[PER (Prioritized Experience Replay)](https://arxiv.org/abs/1511.05952)|[PrioritizedExperienceReplay](https://github.com/kengz/openai_lab/blob/master/rl/memory/prioritized_exp_replay.py)|-|
|[CEM (Cross Entropy Method)](https://en.wikipedia.org/wiki/Cross-entropy_method)|next|-|
|[REINFORCE](http://incompleteideas.net/sutton/williams-92.pdf)|-|-|
|[DPG (Deterministic Policy Gradient) off-policy actor-critic](http://jmlr.org/proceedings/papers/v32/silver14.pdf)|[ActorCritic](https://github.com/kengz/openai_lab/blob/master/rl/agent/actor_critic.py)|-|
|[DDPG (Deep-DPG) actor-critic with target networks](https://arxiv.org/abs/1509.02971)|[DDPG](https://github.com/kengz/openai_lab/blob/master/rl/agent/ddpg.py)|-|
|[A3C (asynchronous advantage actor-critic)](https://arxiv.org/pdf/1602.01783.pdf)|-|-|
|Dyna|next|-|
|[TRPO](https://arxiv.org/abs/1502.05477)|-|-|
|Q*(lambda)|-|-|
|Retrace(lambda)|-|-|
|[Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988)|-|-|
|[EWC (Elastic Weight Consolidation)](https://arxiv.org/abs/1612.00796)|-|-|


### Run the Lab

Next, see [Installation](http://kengz.me/openai_lab/#installation) and jump to [Quickstart](http://kengz.me/openai_lab/#quickstart).


<div style="max-width: 100%"><img alt="Timelapse of OpenAI Lab" src="http://kengz.me/openai_lab/images/lab_demo_dqn.gif" /></div>

*Timelapse of OpenAI Lab, solving CartPole-v0.*
