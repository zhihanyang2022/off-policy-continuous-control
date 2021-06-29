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
    id='car-v0',
    entry_point='domains.car:CarEnv',
    max_episode_steps=160
)

register(
    id='car-concat20-v0',
    entry_point='domains.car:concat20',
    max_episode_steps=160
)

# ============================================================================================
# Pendulum Swing-up Variable Length (Sensor Integration + System Identification)
# ============================================================================================

# with non-recurrent agent: baseline (lstm <= this)
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
# CartPole Balance (for testing recurrent agent with variable length episodes)
# ============================================================================================

register(
    id='cartpole-balance-mdp-v0',
    entry_point='domains.cartpole_balance:mdp',
    max_episode_steps=200,
)

register(
    id='cartpole-balance-pomdp-v0',
    entry_point='domains.cartpole_balance:pomdp',
    max_episode_steps=200,
)

register(
    id='cartpole-balance-mdp-concat5-v0',
    entry_point='domains.cartpole_balance:mdp_concat5',
    max_episode_steps=200,
)

# ============================================================================================
# DMC CartPole Balance (dense reward, fixed episode length)
# ============================================================================================

# these envs are by default last for 1000 / frame_skip
# we use frame_skip of 5, so timeout would be 200

register(
    id='dmc-cartpole-balance-mdp-v0',
    entry_point='domains.dmc_cartpole_b:mdp',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-balance-pomdp-v0',
    entry_point='domains.dmc_cartpole_b:pomdp',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-balance-mdp-concat5-v0',
    entry_point='domains.dmc_cartpole_b:mdp_concat5',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-balance-mdp-img-concat3-v0',
    entry_point='domains.dmc_cartpole_b:mdp_img_concat3',
    max_episode_steps=200
)

# ============================================================================================
# DMC CartPole Swing-up (dense reward, fixed episode length)
# ============================================================================================

# these envs are by default last for 1000 / frame_skip
# we use frame_skip of 5, so timeout would be 200

register(
    id='dmc-cartpole-swingup-mdp-v0',
    entry_point='domains.dmc_cartpole_su:mdp',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-swingup-pomdp-v0',
    entry_point='domains.dmc_cartpole_su:pomdp',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-swingup-mdp-concat5-v0',
    entry_point='domains.dmc_cartpole_su:mdp_concat5',
    max_episode_steps=200
)

# ============================================================================================
# CartPole Swing-up Variable Length (Sensor Integration + System Identification)
# ============================================================================================

# with non-recurrent agent: baseline (lstm <= this)
register(
    id='cartpole-var-len-pvl-v0',
    entry_point='domains.cartpole_var_len:pvl',
    max_episode_steps=500
)

# with non-recurrent agent: baseline (ablation)
register(
    id='cartpole-var-len-pv-v0',
    entry_point='domains.cartpole_var_len:pv',
    max_episode_steps=500
)

# with non-recurrent agent: baseline (lstm = this)
register(
    id='cartpole-var-len-pa-concat5-v0',
    entry_point='domains.cartpole_var_len:pa_concat5',
    max_episode_steps=500
)

# with non-recurrent agent: baseline (lstm > this)
# with recurrent agent: for lstm
register(
    id='cartpole-var-len-pa-v0',
    entry_point='domains.cartpole_var_len:pa',
    max_episode_steps=500
)

# ============================================================================================
# Reacher (Long-term dependency)
# ============================================================================================

register(
    id='reacher-mdp-v0',
    entry_point='domains.reacher:mdp',
    max_episode_steps=50
)

register(
    id='reacher-pomdp-v0',
    entry_point='domains.reacher:pomdp_v0',
    max_episode_steps=50
)

register(
    id='reacher-pomdp-v1',
    entry_point='domains.reacher:pomdp_v1',
    max_episode_steps=50
)

register(
    id='water-maze-mdp-v0',
    entry_point='domains.water_maze:mdp',
    max_episode_steps=200,
)

register(
    id='water-maze-pomdp-v0',
    entry_point='domains.water_maze:pomdp',
    max_episode_steps=200,
)

register(
    id='water-maze-dense-mdp-v0',
    entry_point='domains.water_maze_dense:mdp',
    max_episode_steps=200,
)

register(
    id='water-maze-dense-pomdp-v0',
    entry_point='domains.water_maze_dense:pomdp',
    max_episode_steps=200,
)

# ============================================================================================
# Bumps normal (for HAC paper)
# ============================================================================================

register(
    id='bumps-normal-mdp-v0',
    entry_point='domains.robot_envs.bumps_norm:mdp',
    max_episode_steps=50
)

register(
    id='bumps-normal-pomdp-v0',
    entry_point='domains.robot_envs.bumps_norm:pomdp',
    max_episode_steps=50
)
