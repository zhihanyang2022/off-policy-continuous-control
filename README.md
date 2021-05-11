If use conda, create the env first using conda and then install requirements.txt using pip.

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
