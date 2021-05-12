Utilize gradient clipping, which might improve stability (but I haven't investigated this)

## Commands

### Commands for running and visualizing on classic control tasks

For `cartpole-continuous-v0`:

```bash
python run.py --env cartpole-continuous-v0 --algo ddpg --config configs/classic_control_ddpg.gin --run_id 1 2 3
python run.py --env cartpole-continuous-v0 --algo ddpg --config configs/classic_control_ddpg.gin --run_id 1 2 3 --visualize

python run.py --env cartpole-continuous-v0 --algo td3 --config configs/classic_control_td3.gin --run_id 1 2 3
python run.py --env cartpole-continuous-v0 --algo td3 --config configs/classic_control_td3.gin --run_id 1 2 3 --visualize

python run.py --env cartpole-continuous-v0 --algo sac --config configs/classic_control_sac.gin --run_id 1 2 3
python run.py --env cartpole-continuous-v0 --algo sac --config configs/classic_control_sac.gin --run_id 1 2 3 --visualize
```

<p align="center">
  <img src="results/cartpole-continuous-v0/avg_return.png" width=600>
</p>

For `Pendulum-v0`:

```bash
python run.py --env Pendulum-v0 --algo ddpg --config configs/classic_control_ddpg.gin --run_id 1 2 3
python run.py --env Pendulum-v0 --algo ddpg --config configs/classic_control_ddpg.gin --run_id 1 2 3 --visualize

python run.py --env Pendulum-v0 --algo td3 --config configs/classic_control_td3.gin --run_id 1 2 3
python run.py --env Pendulum-v0 --algo td3 --config configs/classic_control_td3.gin --run_id 1 2 3 --visualize

python run.py --env Pendulum-v0 --algo sac --config configs/classic_control_sac.gin --run_id 1 2 3
python run.py --env Pendulum-v0 --algo sac --config configs/classic_control_sac.gin --run_id 1 2 3 --visualize
```

<p align="center">
  <img src="results/Pendulum-v0/avg_return.png" width=600>
</p>

### Commands for running and visualizing Mujoco tasks

For `Ant-v3`:

```bash
python run.py --env Ant-v3 --algo sac --config configs/mujoco_sac.gin --run_id 1 2 3
python run.py --env Ant-v3 --algo sac --config configs/mujoco_sac.gin --run_id 1 2 3 --visualize
```


Talk about how both SAC and TD3 uses a target policy net, which is not present in SAC

Make sure you have mujoco properly installed at whatever directory
If you get this error after installing requirements.txt:

```bash
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```

simply do `pip uninstall numpy` and `pip install numpy`, which sounds silly but works. Thanks to

https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp

I/ use conda, create the env first using conda and then install requirements.txt using pip.

Comment on where working directly should be

CITE image

PLEASE NOTE THAT THIS REPO IS UNDER ACTIVE CONSTRUCTION.

Table of content

# Off-policy methods for continuous control 🧚‍♂️ 

*Clean, modular implementation of model-free off-policy methods for continuous control in PyTorch.*

*This repo implements DDPG, TD3 and SAC and tests them against OpenAI Spinning Up's baselines.*

*OpenAI Spinning Up helped me tremendously along the way.*

<p align="center">
  <img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" width=600>
</p>

## Features

|              Problems with some repos              |                 Solutions                |
|:--------------------------------------------------:|:--------------------------------------------:|
| "Code works but I don't understand the how"        | Offers docs and implementation notes                 | 
| Uses sophisticated abstraction                     | Offers graphical explanation of design choices |
| Does not have a good requirements file             | Has a requirements file tested across multiple machines |
|    Does not compare against authoritative repos    |       Compares against OpenAI Spinning Up*       |
|         Does not test on many environments         |   Tests on many tasks including Atari & Mujoco |

\* However, not all algorithms here are implemented in OpenAI Spinning Up.

## Codebase design

The diagrams below are created using Lucidchart.

TODO:
- The design of the actor and critic class
  - not meant to be used directly, instantiated within the algorithm
  - Randomness should be handled within algorithm, not actor; even for SAC, the mean and std of Gaussian are outputed by actor, then processed in algorithm

### Overview

<p align="center">
  <img src="diagrams/design.svg" width=600>
</p>

### Abstract classes

## Implemented algorithms and notes

Q-learning algorithms:
- Deep Q-learing
- Categorical 51
- <a target="_blank" href="https://nbviewer.jupyter.org/github/zhihanyang2022/CleanRL/blob/main/notes/qrdqn.pdf" type="application/pdf">Quantile-Regression DQN</a>

Policy optimization algorithms:
- TODO

Both:
- TODO

## Gallery of GIFs of learned policies
