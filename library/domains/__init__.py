from gym.envs.registration import register

# Caution: all envs here must set max_episode_steps!!

register(
    id='cartpole-continuous-v0',
    entry_point='domains.cartpole_continuous:ContinuousCartPoleEnv',
    max_episode_steps=200
)
