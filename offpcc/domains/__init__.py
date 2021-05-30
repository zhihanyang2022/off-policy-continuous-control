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

# ============================================================================================
# Pendulum Swing-up Variable Length (Type 1 Task)
# ============================================================================================

# with non-recurrent agent: baseline (lstm < this)
register(
    id='pendulum-var-len-pvl-v0',
    entry_point='domains.pendulum_var_len:pvl',
    max_episode_steps=200
)

# with non-recurrent agent: baseline (ablation)
register(
    id='pendulum-var-len-pv-v0',
    entry_point='domains.pendulum_var_len:pv',
    max_episode_steps=200
)

# with non-recurrent agent: baseline (ablation)
register(
    id='pendulum-var-len-pl-v0',
    entry_point='domains.pendulum_var_len:pl',
    max_episode_steps=200
)

# with non-recurrent agent: baseline (lstm = this)
register(
    id='pendulum-var-len-pa-concat5-v0',
    entry_point='domains.pendulum_var_len:pa_concat5',
    max_episode_steps=200
)

# with non-recurrent agent: baseline (lstm > this)
# with recurrent agent: for lstm
register(
    id='pendulum-var-len-pa-v0',
    entry_point='domains.pendulum_var_len:pa',
    max_episode_steps=200
)

# ============================================================================================
# CartPole Variable Action Multiplier (Type 1 Task)
# I'm using 350 timesteps because
# 1) it is used in RDPG paper
# 2) it is harder than 200 timesteps (in gym) but easier than 1000 timesteps (in VRM)
# ============================================================================================

# with non-recurrent agent: baseline (lstm < this)
register(
    id='cartpole-vam-v0',
    entry_point='domains.cartpole_continuous_variable_action_multiplier:ContinuousCartPoleVariableActionMultiplierEnv',
    max_episode_steps=350
)

# with non-recurrent agent: baseline (prove that our task is really harder)
register(
    id='cartpole-vam-pv-v0',
    entry_point='domains.cartpole_continuous_variable_action_multiplier:pv',
    max_episode_steps=350
)

# with non-recurrent agent: baseline (lstm approximately = this)
register(
    id='cartpole-vam-p-concat10-v0',
    entry_point='domains.cartpole_continuous_variable_action_multiplier:p_concat10',
    max_episode_steps=350
)

# with non-recurrent agent: baseline (lstm > this)
# with recurrent agent: for lstm
register(
    id='cartpole-vam-p-v0',
    entry_point='domains.cartpole_continuous_variable_action_multiplier:p',
    max_episode_steps=350
)

register(
    id='water-maze-v0',
    entry_point='domains.water_maze:WaterMazeEnv',
    max_episode_steps=200,
)