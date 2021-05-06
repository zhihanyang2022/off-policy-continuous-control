# CleanRL 🧚‍♂️ 

*Clean implementation of **model-free** deep RL algorithms using both **PyTorch** and **TensorFlow**.*

<p align="center">
  <img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" width=600>
</p>

[CITE image]

*This repo has a strong focus on helping learners.*

*Inpsired by OpenAI Spinning Up, which helped me tremendously along the way.*

PLEASE NOTE THAT THIS REPO IS UNDER ACTIVE CONSTRUCTION.

## Manifesto

|              Problems with some repos              |                 Solutions                |
|:--------------------------------------------------:|:--------------------------------------------:|
| "Code works but I don't understand the how"       | Offers docs and implementation notes                 | 
| Uses sophisticated abstraction                     | Offers graphical explanation of design choices |
|                  Uses one library                  |       Uses both PyTorch and TensorFlow       |
| Does not have a good requirements file             | Has a requirements file tested across multiple machines |
|    Does not compare against authoritative repos    |       Compares against OpenAI Spinning Up*       |
|         Does not test on many environments         |   Tests on many tasks including Mujoco ones  |

\* However, not all algorithms here are implemented in OpenAI Spinning Up.

## Design choices

## Implemented algorithms and notes

Q-learning algorithms:
- <a target="_blank" href="https://nbviewer.jupyter.org/github/zhihanyang2022/CleanRL/blob/main/notes/qrdqn.pdf" type="application/pdf">Quantile-Regression DQN</a>
