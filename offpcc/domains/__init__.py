from gym.envs.registration import register

# car heaven hell

register(
    id='car-v0',
    entry_point='domains.car:CarEnv',
    max_episode_steps=160
)


register(
    id='car-episodic-v0',
    entry_point='domains.car_episodic:CarEnv',
    max_episode_steps=160
)

register(
    id='car-concat20-v0',
    entry_point='domains.car:concat20',
    max_episode_steps=160
)

register(
    id='car-episodic-concat20-v0',
    entry_point='domains.car_episodic:concat20',
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
    id='dmc-cartpole-balance-v0',
    entry_point='domains.dmc_cartpole_b:mdp',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-balance-p-v0',
    entry_point='domains.dmc_cartpole_b:p',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-balance-va-v0',
    entry_point='domains.dmc_cartpole_b:va',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-balance-p-concat5-v0',
    entry_point='domains.dmc_cartpole_b:p_concat5',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-balance-va-concat10-v0',
    entry_point='domains.dmc_cartpole_b:va_concat10',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-swingup-v0',
    entry_point='domains.dmc_cartpole_su:mdp',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-swingup-p-v0',
    entry_point='domains.dmc_cartpole_su:p',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-swingup-va-v0',
    entry_point='domains.dmc_cartpole_su:va',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-swingup-p-concat5-v0',
    entry_point='domains.dmc_cartpole_su:p_concat5',
    max_episode_steps=200
)

register(
    id='dmc-cartpole-swingup-va-concat10-v0',
    entry_point='domains.dmc_cartpole_su:va_concat10',
    max_episode_steps=200
)

register(
    id='dmc-cart2pole-v0',
    entry_point='domains.dmc_cart2pole:mdp',
    max_episode_steps=200
)

register(
    id='dmc-cart3pole-v0',
    entry_point='domains.dmc_cart3pole:mdp',
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

register(
    id='water-maze-simple-mdp-v0',
    entry_point='domains.water_maze_simple:mdp',
    max_episode_steps=200,
)

register(
    id='water-maze-simple-pomdp-v0',
    entry_point='domains.water_maze_simple:pomdp',
    max_episode_steps=200,
)

register(
    id='water-maze-simple-mdp-concat10-v0',
    entry_point='domains.water_maze_simple:mdp_concat10',
    max_episode_steps=200,
)

# ============================================================================================
# robot envs (for HAC paper)
# ============================================================================================

register(
    id='pbc-bumps-normal-pomdp-v0',
    entry_point='domains.robot_envs.bumps_norm:BumpsNormEnv',
    max_episode_steps=50
)

register(
    id='pbc-bumps-normal-pomdp-punish-v0',
    entry_point='domains.robot_envs.bumps_norm_punish:BumpsNormEnv',
    max_episode_steps=50
)

register(
    id='pbc-bumps-normal-test-v0',
    entry_point='domains.robot_envs.bumps_norm_test:BumpsNormEnv',
    max_episode_steps=50
)

register(
    id='pbc-bumps-normal-pomdp-real-v0',
    entry_point='domains.robot_envs.bumps_norm_real:BumpsNormEnv',
    max_episode_steps=50
)

register(
    id='pbc-bump-target-pomdp-v0',
    entry_point='domains.robot_envs.bump_target:BumpTargetEnv',
    max_episode_steps=50
)

register(
    id='pbc-bump-mdp-v0',
    entry_point='domains.bump_mdp:BumpEnv',
    max_episode_steps=200
)

# ============================================================================================
# dm control's image-based envs (for verifying our convolutional sac implementation)
# ============================================================================================

register(
    id='dmc-cartpole-balance-mdp-img-concat3-v0',
    entry_point='domains.dmc_cartpole_b:mdp_img_concat3',
    max_episode_steps=500
)

register(
    id='dmc-walker-walk-mdp-img-concat3-v0',
    entry_point='domains.dmc_walker_walk:mdp_img_concat3',
    max_episode_steps=500
)

# ============================================================================================
# for comparison with VRM
# ============================================================================================

register(
    id='pendulum-v0',
    entry_point='domains.pendulum_swingup_from_vrm:mdp',
    max_episode_steps=200
)

register(
    id='pendulum-p-v0',
    entry_point='domains.pendulum_swingup_from_vrm:p',
    max_episode_steps=200
)

register(
    id='pendulum-v-v0',
    entry_point='domains.pendulum_swingup_from_vrm:v',
    max_episode_steps=200
)

register(
    id='pendulum-va-v0',
    entry_point='domains.pendulum_swingup_from_vrm:va',
    max_episode_steps=200
)

register(
    id='pendulum-p-concat5-v0',
    entry_point='domains.pendulum_swingup_from_vrm:p_concat5',
    max_episode_steps=200
)

register(
    id='pendulum-v-concat10-v0',
    entry_point='domains.pendulum_swingup_from_vrm:v_concat10',
    max_episode_steps=200
)

register(
    id='pendulum-va-concat10-v0',
    entry_point='domains.pendulum_swingup_from_vrm:va_concat10',
    max_episode_steps=200
)

register(
    id='cartpole-continuous-long-v0',
    entry_point='domains.cartpole_balance:mdp',
    max_episode_steps=1000
)

register(
    id='cartpole-continuous-long-p-v0',
    entry_point='domains.cartpole_balance:p',
    max_episode_steps=1000
)

register(
    id='cartpole-continuous-long-v-v0',
    entry_point='domains.cartpole_balance:v',
    max_episode_steps=1000
)

register(
    id='cartpole-continuous-long-p-concat5-v0',
    entry_point='domains.cartpole_balance:p_concat5',
    max_episode_steps=1000
)

register(
    id='cartpole-continuous-long-v-concat10-v0',
    entry_point='domains.cartpole_balance:v_concat10',
    max_episode_steps=1000
)

# ============================================================================================
# for ICRA
# ============================================================================================

register(
    id='car-top-v0',
    entry_point='domains.car_top:CarEnv',
    max_episode_steps=10
)

register(
    id='car-top-relative-v0',
    entry_point='domains.car_top_relative:CarEnv',
    max_episode_steps=10
)

register(
    id='car-top-v1',
    entry_point='domains.car_top_narrow:CarEnv',
    max_episode_steps=10
)

register(
    id='ant-reacher-top-v0',
    entry_point='domains.ant_reacher_top:AntEnv',
    max_episode_steps=20
)

register(
    id='bump-top-v0',
    entry_point='domains.bump_top:BumpEnv',
    max_episode_steps=20
)

register(
    id='box-top-v0',
    entry_point='domains.box_top:BoxEnv',
    max_episode_steps=20
)

register(
    id='ur5-top-v0',
    entry_point='domains.ur5_top:Ur5Env',
    max_episode_steps=200
)

register(
    id='ur5-mdp-top-v0',
    entry_point='domains.ur5_mdp_top:Ur5Env',
    max_episode_steps=10
)

register(
    id='halfcheetah-p-v0',
    entry_point='domains.pybullet_halfcheetah:p',
    max_episode_steps=1000
)

register(
    id='ant-p-v0',
    entry_point='domains.pybullet_ant:p',
    max_episode_steps=1000
)

register(
    id='dmc-pendulum-swingup-v0',
    entry_point='domains.dmc_pendulum_su:mdp',
    max_episode_steps=200
)
