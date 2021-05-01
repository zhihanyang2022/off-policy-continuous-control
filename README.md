# clean-rl
Implementation of important Deep RL algorithms using both PyTorch and TF, with a strong focus on understandable code.

- Model-free
  - Policy optimization (only)
  - Q-learning and policy optimization
    - Soft actor critic
  - Q-learning (only)
- Model-based (TODO)

**Problems** with some other implementations and our *solutions*:
- **Uses sophisticated abstraction without explanation.** *Offers graphical explanation of design choices`
- **Uses only one library.** Only offer implementation in PyTorch or TensorFlow. In addition, requirements are not clearly specified and this leads to frustration in getting the repository to do a single run.
- **Does not compare with other repos.** Does not make comparison with mroe authoritative and more popular RL codebases that tend to be used a lot (more likely to be correct)
- **Does not test on many environments.** 
