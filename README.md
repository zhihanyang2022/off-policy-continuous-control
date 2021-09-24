![offpcc_logo](https://user-images.githubusercontent.com/43589364/132990408-91d68fa7-5bed-4298-b554-da6da4c80fd2.png)

## Introduction

This PyTorch repo implements off-policy RL algorithms for continuous control, including:

-   Standard algorithms: DDPG, TD3, SAC
-   Image-based algorithm: ConvolutionalSAC
-   Recurrent algorithms: RecurrentDPG, RecurrentTD3, RecurrentSAC, RecurrentSACNoSharing (see report)

where recurrent algorithms are generally not available in other repos.

Several good things about this repo:

-   Code is readable and extendable; all algorithms share the same class template.
-   Code is easily configurable using gin-configs. We provide a short tutorial on gin-config (see Extra)
-   We describe the structure of the codebase so you don’t have to figure it out yourself (see Codebase structure).
-   We benchmark standard algorithms on several Pybullet tasks against stable-baselines3 or SB3 (see Standard algorithms). We provide an interface to SB3 (see Extra) so that our code and SB3’s code can use the same configs, but this only applies to standard algorithms.
-   We augment ConvolutionalSAC with Data Regularized Q to reproduce results in that paper (see Image-based algorithm).
-   Design of recurrent agents and their results across various domains are available in the following technical report. We also provide the commands for reproducing these results in the report (see Recurrent algorithms).

If you use this repo for your research, consider citing the technical report:

[cite the technical report]

We welcome your feedback and questions through Issues.

## Structure of codebase

Here, we talk about the organization of this code. In particular, we will talk about

-   Folder: where are certain files located?
-   Classes: how are classes designed to interact with each other?
-   Training/evaluation loop: how environment interaction, learning and evaluation alternate?

A basic understanding of these will make other details easy to understand from code itself.

### Folders

-   file
    -   containing plots reproducing stable-baselines3; you don’t need to touch this
-   offpcc (the good stuff; you will be using this)
    -   algorithms (where DDPG, TD3 and SAC are implemented)
    -   algorithms_recurrent (where RDPG, RTD3 and RSAC are implemented)
    -   basics (abstract classes, stuff shared by algorithms or algorithms_recurrent, code for training)
    -   basics_sb3 (you don’t need to touch this)
    -   configs (gin configs)
    -   domains (all custom domains are stored within and registered properly)
-   pics_for_readme
    -   random pics; you don’t need to touch this
-   temp
    -   potentially outdated stuff; you don’t need to touch this

### Relationships between classes

There are three core classes in this repo:

-   Any environment written using OpenAI’s API would have:
    -   `reset` method outputs the current state
    -   `step` method takes in an action, outputs (reward, next state, done, info)
-   `OffPolicyRLAlgorithm` and `RecurrentOffPolicyRLAlgorithm` are the base class for all algorithms listed in introduction. You should think about them as neural network (e.g., actors, critics, CNNs, RNNs) wrappers that are augmented with methods to help these networks interact with other stuff:
    -   `act` method takes in state from env, outputs action back to env
    -   `update_networks` method takes in batch from buffer
-   The replay buffers `ReplayBuffer` and `RecurrentReplayBuffer` are built to interact with the environment and the algorithm classes
    -   `push` method takes in a transition from env
    -   `sample` method outputs a batch for algorithm’s `update_networks` method

Their relationships are best illustrated by a diagram:

![offpcc_steps](https://user-images.githubusercontent.com/43589364/132971785-03d345a0-cef9-484c-bad9-79174d905269.jpg)

### Structure of training/evaluation loop

In this repo, we follow the training/evaluation loop style in spinning-up (this is essentially the script: `basics/run_fns` and the function `train`). It follows this basic structure, with details added for tracking stats and etc:

```python
state = env.reset()
for t range(total_steps):  # e.g., 1 million
    # environment interaction
    if t >= update_after:
        action = algorithm.act(state)
    else:
        action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
   	# learning
    if t >= update_after and (t + 1) % update_every == 0:
        for j in range(update_every):
            batch = buffer.sample()
            algorithm.update_networks(batch)
    # evaluation
    if (t + 1) % num_steps_per_epoch == 0:
        ep_len, ep_ret = test_for_one_episode(test_env, algorithm)
```

## Dependencies

Blabla

The othermost env wrapper should always be a timelimit wrapper (mention code reasons)

## Train an agent

### Setup (wandb & GPU)

Add this to your bashrc or bash_profile and source it.

You should replace “pomdpr” with whatever wandb account that you want to use.

```
export OFFPCC_WANDB_ENTITY="pomdpr"
```

From the command line:

```
cd offpcc
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=project123 python launch.py --env <env-name> --algo <algo-name> --config <config-path> --run_id <id>
```

### Command line template

```bash
python launch.py --env ENV_NAME --algo ALGO_NAME --config CONFIG_PATH --run_id RUN_ID
```

There are multiple options for each field. 

-   `ENV_NAME` can be:
    -   A environment from OpenAI gym, e.g., `pendulum-v0`. Mujoco environments are directly obtainable from gym if the corresponding packages like `mujoco_py` are installed, so they are included in this category.
    -   A Pybullet env. The `pybullet_envs` library is imported at the top of `launch.py`
    -   A custom environment stored in `domains`. The instructions for adding your own custom domain is available in the Extra section.
-   `ALGO_NAME` need to satisfy some requirements:
    -   Must be included in the keys of the `algo_name2class` directionary in `launch.py`.
    -   Non-image envs CANNOT be used with convolutional algorithms. 
    -   Image-based envs CANNOT be used with standard or recurrent algorithms.
-   `CONFIG_PATH` is the relative path of a config file inside `configs`. Again, you CANNOT use an arbitrary config with an arbitrary algorithm, you need to make sure that the specified config ACTUALLY configures all classes and objects in the call hierarchy. Here are some template configs already in the repo: 
    -   `configs/test/template_short.gin`: for DDPG, TD3 and SAC; for 200k steps
    -   `configs/test/template_cnn.gin`: for ConvolutionalSAC; for 1 million steps; with DRQ
    -   `configs/test/template_cnn_noaug.gin`: for ConvolutionalSAC; for 1 million steps; without DRQ
    -   `configs/test/recurrent.gin`: for RecurrentDDPG, RecurrentTD3 and RecurrentSAC; for 200k steps
    -   `configs/test/recurrent.gin`: for RecurrentDDPG, RecurrentTD3 and RecurrentSAC; for 1 million steps
-   `RUN_ID` is simply a integer assigned to a specific run; it’s NOT a seed. In fact, in this repo, we don’t use seeding at all. To run multiple runs (e.g., 5 runs), simply pass in multiple run ids, e.g., `--run_id 1 2 3 4 5`. 

Training stats are available from 3 places:

-   Directly within console
-   Within the corresponding wandb project and the corresponding run group
-   After the run, the csv for the run can be downloaded from wandb

There are 3 types of training stats logged on wandb:

-   Performance: Episode length (train and test), episode return (train and test)
-   Debugging: Q-values, losses (entropy coefficient for SAC-based algorithms)
-   Hours elapsed

Below, we give one simple example for each category of algorithms.

### For DDPG, TD3, SAC

On `Pendulum-v0` (from gym):

```bash
python launch.py --env Pendulum-v0 --algo sac --config configs/test/template_short.gin --run_id 1
```

Commands and plots for benchmarking on Pybullet domains are in a Issue called “Performance check against SB3”.

### For ConvolutionalSAC

On `dmc-cartpole-balance-mdp-img-concat3-v0` (custom env; concatenating 3 images to infer velocity):

```bash
python launch.py --env dmc-cartpole-balance-mdp-img-concat3-v0 --algo csac --config configs/test/template_cnn.gin --run_id 1
```

### For RecurrentDDPG, RecurrentTD3, RecurrentSAC

On `Pendulum-p-v0` (custom env; position only)

```bash
python launch.py --env pendulum-p-v0 --algo rsac --config configs/test/template_recurrent_100k.gin --run_id 1
```

## Visualize a trained policy

After training is done, the policy will be uploaded to wandb, and you can find them here. Simply click download at the very right. Then, within off-policy-continuous-control but outside offpcc, create the following folder structure:

```
results/pbc-bumps-normal-pomdp-v0/rsac/1/
```

and move the downloaded stuff within.

As a sidenode, for recurrent agents, we store the policy as two parts: `actor.pth` (the mlp part) and `actor_summarizer.pth` (the lstm part). Of course, the full policy needs both component to work together.

After you’ve put the trained networks in the right place, simply run the following command, but make sure that you temporarily change the render argument in the init method of robot envs, otherwise nothing will be shown. Along the visualization, some testing stats will be printed (e.g., lengths of trajectories, success)

```
python launch.py --env pbc-bumps-normal-pomdp-v0 --algo rsac --config configs/test/template_recurrent_100k.gin --run_id 1 --render
```

![Screen Shot 2021-07-22 at 11.49.35 AM](https://i.loli.net/2021/07/22/OckBDTZXqxbfS1C.png)

## Reproduce results of DrQ

TODO

## Reproduce results of technical report

### Experiment 1: comparing base algorithm

Project name: report-pendulum

MDP baseline

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-v0 --algo sac --config configs/test/template_100k.gin --run_id 1 2 3 4
```

`pendulum-p-v0` (gym)

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-p-v0 --algo sac --config configs/test/template_short.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-p-v0 --algo rdpg --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-p-v0 --algo rtd3 --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-p-v0 --algo rsac --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4
```

`pendulum-v-v0` (gym)

```bash
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-v-v0 --algo sac --config configs/test/template_short.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-v-v0 --algo rdpg --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-v-v0 --algo rtd3 --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-v-v0 --algo rsac --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4
```

`pendulum-va-v0 `(gym)

```bash
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-va-v0 --algo sac --config configs/test/template_short.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-va-v0 --algo rdpg --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-va-v0 --algo rtd3 --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-pendulum python launch.py --env pendulum-va-v0 --algo rsac --config configs/test/template_recurrent_100k.gin --run_id 1 2 3 4
```

dmc-cartpole-balance-v0

```bash
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&
```

dmc-cartpole-balance-p-v0

```bash
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-p-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-p-concat5-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-p-v0 --algo rdpg --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-p-v0 --algo rtd3 --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-p-v0 --algo rsac --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4
```

dmc-cartpole-balance-va-v0

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-va-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-va-concat10-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-va-v0 --algo rdpg --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-va-v0 --algo rtd3 --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-cartpole-balance python launch.py --env dmc-cartpole-balance-va-v0 --algo rsac --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4
```

dmc-cartpole-swingup-v0

```bash
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&
```

dmc-cartpole-swingup-p-v0

```bash
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-p-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-p-concat5-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-p-v0 --algo rdpg --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-p-v0 --algo rtd3 --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-p-v0 --algo rsac --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4
```

dmc-cartpole-swingup-va-v0

```bash
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-va-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-va-concat10-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-va-v0 --algo rdpg --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-va-v0 --algo rtd3 --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-cartpole-swingup python launch.py --env dmc-cartpole-swingup-va-v0 --algo rsac --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4
```

todo:

-   check the domains
-   consider the next two environments to test

Reacher

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-reacher python launch.py --env reacher-mdp-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-reacher python launch.py --env reacher-pomdp-v0 --algo sac --config configs/test/template_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-reacher python launch.py --env reacher-pomdp-v0 --algo rsac --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4

CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-reacher python launch.py --env reacher-pomdp-v0 --algo rdpg --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-reacher python launch.py --env reacher-pomdp-v0 --algo rtd3 --config configs/test/template_recurrent_200k.gin --run_id 1 2 3 4
```

Watermaze

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-mdp-v0 --algo sac --config configs/test/template_500k.gin --run_id 1 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo sac --config configs/test/template_500k.gin --run_id 1
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rsac --config configs/test/template_recurrent_1m.gin --run_id 1
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rsac --config configs/test/template_recurrent_1m.gin --run_id 2
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rsac --config configs/test/template_recurrent_1m.gin --run_id 3
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rsac --config configs/test/template_recurrent_500k.gin --run_id 4
```

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rtd3 --config configs/test/template_recurrent_500k.gin --run_id 1 &&\
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rtd3 --config configs/test/template_recurrent_500k.gin --run_id 2
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rtd3 --config configs/test/template_recurrent_500k.gin --run_id 3 &&\
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rtd3 --config configs/test/template_recurrent_500k.gin --run_id 4
```

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rdpg --config configs/test/template_recurrent_500k.gin --run_id 1
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rdpg --config configs/test/template_recurrent_500k.gin --run_id 2
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rdpg --config configs/test/template_recurrent_500k.gin --run_id 3
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-watermaze python launch.py --env water-maze-simple-pomdp-v0 --algo rdpg --config configs/test/template_recurrent_500k.gin --run_id 4
```

**bumps-normal-test**

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-bumps-normal-test python launch.py --env pbc-bumps-normal-test-v0 --algo sac --config configs/test/template_500k.gin --run_id 1 2 3 4
```

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-bumps-normal-test python launch.py --env pbc-bumps-normal-test-v0 --algo rdpg --config configs/test/template_recurrent_500k.gin --run_id 1 2 3
```

```bash
CUDA_VISIBLE_DEVICES=1 OFFPCC_WANDB_PROJECT=report-bumps-normal-test python launch.py --env pbc-bumps-normal-test-v0 --algo rtd3 --config configs/test/template_recurrent_500k.gin --run_id 1 2 3
```

```bash
CUDA_VISIBLE_DEVICES=2 OFFPCC_WANDB_PROJECT=report-bumps-normal-test python launch.py --env pbc-bumps-normal-test-v0 --algo rsac --config configs/test/template_recurrent_500k.gin --run_id 1 2 3
```

```bash
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-bumps-normal-test python launch.py --env pbc-bumps-normal-test-v0 --algo rdpg --config configs/test/template_recurrent_500k.gin --run_id 4 &&\
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-bumps-normal-test python launch.py --env pbc-bumps-normal-test-v0 --algo rtd3 --config configs/test/template_recurrent_500k.gin --run_id 4 &&\
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-bumps-normal-test python launch.py --env pbc-bumps-normal-test-v0 --algo rsac --config configs/test/template_recurrent_500k.gin --run_id 4
```

**halfcheetah-p-v0**

```bash
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-halfcheetah python launch.py --env HalfCheetahBulletEnv-v0 --algo sac --config configs/test/template_1m_pybullet.gin --run_id 1
CUDA_VISIBLE_DEVICES=0 OFFPCC_WANDB_PROJECT=report-halfcheetah python launch.py --env halfcheetah-p-v0 --algo sac --config configs/test/template_1m_pybullet.gin --run_id 1
```

```bash
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=report-halfcheetah python launch.py --env halfcheetah-p-v0 --algo rsac --config configs/test/template_recurrent_1m_pybullet.gin --run_id 1
```






#### How to add a custom domain

#### talk a little bit about how to confiure the classes using gin-config

TODO: commands for running, a bit more about what happens behind the scene,, domains available

Here's an example of running bumps norm:

```
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=bumps-norm-recurrent python launch.py --env pbc-bumps-normal-pomdp-v0 --algo rsac --config configs/test/template_recurrent_100k.gin --run_id 1
```

Breaking it down:

-   `CUDA_VISIBLE_DEVICES=3no`: Running recurrent agents can be computationally expensive for GPU. Therefore, before running anything, do check by `nvidia-smi` that no one is using the GPU you want to run on.
-   `pbc-bumps-normal-pomdp-v0`: I register this env in `domains/__init__.py`. The prefix `pbc` stands for py-bullet-custom, i.e., pybullet envs created by ourselves. These envs have a huge problem. You cannot simultaneously create 2 versions of the env (one for training and one for testing), otherwise 4 bumps would show up in the same playground, which totally destroys the env. Therefore, whenever an env has `pbc` as prefix, we do not do testing and just report training stats.
-   `rsac`: It can be `rdpg` or `rtd3` as well.
-   `configs/config/test/template_recurrent_100k.gin`: You can look into the config to get information about buffer and number of training episodes and etc.
-   `run_id`: This is not a seed. In fact, I avoid using seeds because I would be averaging over multiple seeds anyway. This is only an identifier and will be added to wandb to differentiate between runs. Must be int.
