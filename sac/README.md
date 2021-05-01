How to run OpenAI Spinningup baseline:

```python
spinup.sac_pytorch(
    env_fn, 
    actor_critic=<MagicMock spec='str' id='140554319922904'>, 
    ac_kwargs={}, 
    seed=0, 
    steps_per_epoch=10000,  # from benchmark page 
    epochs=300,  # from benchmark page 
    replay_size=1000000, 
    gamma=0.99, 
    polyak=0.995, 
    lr=0.001, 
    alpha=0.2, 
    batch_size=100, 
    start_steps=10000, 
    update_after=1000, 
    update_every=50, 
    num_test_episodes=10,  # from benchmark page
    max_ep_len=1000, 
    logger_kwargs={}, 
    save_freq=1
)
```
