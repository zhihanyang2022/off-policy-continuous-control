from gym.envs.registration import register

# Caution: all envs here must set max_episode_steps!!

register(
    id='cartpole-continuous-v0',
    entry_point='domains.cartpole_continuous:ContinuousCartPoleEnv',
    max_episode_steps=200
)

register(
    id='cartpole-continuous-long-v0',
    entry_point='domains.cartpole_continuous:ContinuousCartPoleEnv',
    max_episode_steps=1000
)

# partially observable: position only

register(
    id='cartpole-continuous-p-v0',
    entry_point='domains.cartpole_continuous:ContinuousCartPoleEnvPositionOnly',
    max_episode_steps=200
)

register(
    id='cartpole-continuous-long-p-v0',
    entry_point='domains.cartpole_continuous:ContinuousCartPoleEnvPositionOnly',
    max_episode_steps=1000
)

# partially observable: velocity only

register(
    id='cartpole-continuous-v-v0',
    entry_point='domains.cartpole_continuous:ContinuousCartPoleEnvVelocityOnly',
    max_episode_steps=200
)

register(
    id='cartpole-continuous-long-v-v0',
    entry_point='domains.cartpole_continuous:ContinuousCartPoleEnvVelocityOnly',
    max_episode_steps=1000
)