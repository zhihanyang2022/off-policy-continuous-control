# Soft actor critic

## Results

## 🚀 Reproduce OpenAI SpinningUp results

### How to run?

PyTorch version:

```bash
python -m spinup.run sac 
    --exp_name sac-pytorch 
    --env [ENV] 
    --steps_per_epoch 10000
    --epochs 300
    --seed [SEED]
```

TensorFlow version:

TODO

Below are the default parameters (before we overwrite them using the command above)

PyTorch version:

```python
spinup.sac_pytorch(
    env_fn, 
    actor_critic=<MagicMock spec='str' id='140554319922904'>, 
    ac_kwargs={}, 
    seed=0, 
    steps_per_epoch=4000, 
    epochs=100,
    replay_size=1000000, 
    gamma=0.99, 
    polyak=0.995, 
    lr=0.001, 
    alpha=0.2, 
    batch_size=100, 
    start_steps=10000, 
    update_after=1000, 
    update_every=50, 
    num_test_episodes=10,
    max_ep_len=1000, 
    logger_kwargs={}, 
    save_freq=1
)
```

TensorFlow version:

TODO

### Where are training logs located?

Training logs are located in `spinningup/data/sac`. Each seed is saved as a separate folder in this directory. For example, `spinningup/data/sac/sac_s0` represents the training logs for SAC with seed 0.

**You should know this.** If you give two runs the same experiment name, then they will share the same logging directory, even if one uses the PyTorch version and the other uses the TensorFlow version of the algorithm.

### How to parse and plot training logs?

Use `clean-rl/sac/plotting_for_spinnup.ipynb`.

## 🚀 Reproduce CleanRL results

### How to run?

### Where are the training logs located?

### How to parse and plot training logs?

Use `clean-rl/sac/plotting_for_cleanrl.ipynb`.

