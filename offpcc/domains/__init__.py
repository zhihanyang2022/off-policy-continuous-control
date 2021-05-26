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
    id='cartpole-continuous-long-p-v0',
    entry_point='domains.cartpole_continuous:continuous_cartpole_position_env',
    max_episode_steps=1000
)

register(
    id='cartpole-continuous-long-p-concat-v0',  # concat obs to approximate MDP
    entry_point='domains.cartpole_continuous:continuous_cartpole_position_concat_env',
    max_episode_steps=1000
)

# partially observable: velocity only

register(
    id='cartpole-continuous-long-v-v0',
    entry_point='domains.cartpole_continuous:continuous_cartpole_velocity_env',
    max_episode_steps=1000
)

register(
    id='cartpole-continuous-long-v-concat-v0',  # concat obs to approximate MDP
    entry_point='domains.cartpole_continuous:continuous_cartpole_velocity_concat_env',
    max_episode_steps=1000
)

# car heaven hell

register(
    id='car-flag-v0',
    entry_point='domains.car:CarEnv',
    max_episode_steps=160
)

register(
    id='car-flag-concat-v0',
    entry_point='domains.car:car_flag_concat',
    max_episode_steps=160
)