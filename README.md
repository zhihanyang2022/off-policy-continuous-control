# clean-rl
Implementation of important Deep RL algorithms using both PyTorch and TF, with a strong focus on understandable code.

- Model-free
  - Policy optimization (only)
  - Q-learning and policy optimization
    - Soft actor critic
  - Q-learning (only)
- Model-based (TODO)

## Why this repo?

|              Problems with some repos              |                 Our solutions                |
|:--------------------------------------------------:|:--------------------------------------------:|
| Uses sophisticated abstraction without explanation | Uses graphical explanation of design choices |
|                  Uses one library                  |       Uses both PyTorch and TensorFlow       |
|    Does not compare against authoritative repos    |       Compares with OpenAI Spinning Up       |
|         Does not test on many environments         |   Tests on many tasks including Mujoco ones  |
