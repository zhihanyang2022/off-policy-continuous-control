# CleanRL

*Clean implementation of important Deep RL algorithms using both PyTorch and TensorFlow*

*It has a strong focus on helping learners.*

PLEASE NOTE THAT THIS REPO IS UNDER ACTIVE CONSTRUCTION.

## Why this repo?

|              Problems with some repos              |                 Our solutions                |
|:--------------------------------------------------:|:--------------------------------------------:|
| "Code works but I don't understand the math"       | Offers implementation notes                  | 
| Uses sophisticated abstraction                     | Offers graphical explanation of design choices |
|                  Uses one library                  |       Uses both PyTorch and TensorFlow       |
| Does not have a good requirements file             | Has a requirements file tested across multiple machines |
|    Does not compare against authoritative repos    |       Compares against OpenAI Spinning Up*       |
|         Does not test on many environments         |   Tests on many tasks including Mujoco ones  |

\* However, not all algorithms here are implemented in OpenAI Spinning Up.

## Implemented algorithms and notes

Q-learning algorithms:
- <a target="_blank" href="https://nbviewer.jupyter.org/github/zhihanyang2022/CleanRL/blob/main/notes/qrdqn.pdf" type="application/pdf">Quantile-Regression DQN</a>
