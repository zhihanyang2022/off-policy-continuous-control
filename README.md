### off-policy-continuous-control

This repo is the official codebase of our following paper:

```
@article{yang2021recurrent,
  title={Recurrent Off-policy Baselines for Memory-based Continuous Control},
  author={Yang, Zhihan and Nguyen, Hai},
  journal={Deep RL Workshop, NeurIPS 2021},
  year={2021}
}
```

Paper summary: We implement and benchmark recurrent versions of DDPG, TD3 and SAC that uses full history.

This repo offers:

- DDPG, TD3 and SAC (clean PyTorch implementation and benchmarked against stable-baselines3*)
- Recurrent versions of DDPG, TD3 and SAC that use full history: RDPG, RTD3 and RSAC
- Very easy to understand and use; see our exhaustive documentation: [link](https://drive.google.com/drive/folders/1iUy5BslSN4zia7VqxqyRnSda4lJO0thV?usp=sharing)

\*The results of benchmarking can be found in issue "Performance check against SB3" in closed Issues.

When cloning this repo, please consider using shallow clone as it is large due to a large number of commits.

Please feel free to ask a code question through Issues.
