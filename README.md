![offpcc_logo](https://user-images.githubusercontent.com/43589364/132990408-91d68fa7-5bed-4298-b554-da6da4c80fd2.png)

TODO: arXiv technical report soon available.

Please ask any questions in Issues, thanks.

## Introduction

This PyTorch repo implements off-policy RL algorithms for continuous control, including:

-   Standard algorithms: DDPG, TD3, SAC
-   Image-based algorithm: ConvolutionalSAC
-   Recurrent algorithms: RecurrentDPG, RecurrentTD3, RecurrentSAC, RecurrentSACSharing (see report)

where recurrent algorithms are generally not available in other repos.

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

Dependencies are available in `requirements.txt`; although there might be one or two missing dependencies that you need to install yourself.

## Train an agent

### Setup (wandb & GPU)

Add this to your bashrc or bash_profile and source it.

You should replace “account_name” with whatever wandb account that you want to use.

```
export OFFPCC_WANDB_ENTITY="account_name"
```

From the command line:

```
cd offpcc
CUDA_VISIBLE_DEVICES=3 OFFPCC_WANDB_PROJECT=project123 python launch.py --env <env-name> --algo <algo-name> --config <config-path> --run_id <id>
```

### For DDPG, TD3, SAC

On `pendulum-v0`:

```bash
python launch.py --env pendulum-v0 --algo sac --config configs/test/template_short.gin --run_id 1
```

Commands and plots for benchmarking on Pybullet domains are in a Issue called “Performance check against SB3”.

### For RecurrentDDPG, RecurrentTD3, RecurrentSAC

On `pendulum-p-v0`:

```bash
python launch.py --env pendulum-p-v0 --algo rsac --config configs/test/template_recurrent_100k.gin --run_id 1
```

## Reproducing paper results

To reproduce paper results, simply run the commands in the previous section with the appropriate env name (listed below) and config files (their file names are highly readable). Mapping between env names used in code and env names used in paper:

-   `pendulum-v0`: `pendulum`
-   `pendulum-p-v0`: `pendulum-p`
-   `pendulum-va-v0`: `pendulum-v`
-   `dmc-cartpole-balance-v0`: `cartpole-balance`
-   `dmc-cartpole-balance-p-v0`: `cartpole-balance-p`
-   `dmc-cartpole-balance-va-v0`: `cartpole-balance-v`
-   `dmc-cartpole-swingup-v0`: `cartpole-swingup`
-   `dmc-cartpole-swingup-p-v0`: `cartpole-swingup-p`
-   `dmc-cartpole-swingup-va-v0`: `cartpole-swingup-v`
-   `reacher-pomdp-v0`: `reacher-pomdp`
-   `water-maze-simple-pomdp-v0`: `watermaze`
-   `bumps-normal-test-v0`: `push-r-bump`

## Render learned policy

Create a folder in the same directory as `offpcc`, called `results`. In there, create a folder with the name of the environment, e.g., `pendulum-p-v0`. Within that env folder, create a folder with the name of the algorithm, e.g., `rsac`. You can get an idea of the algorithms available from the `algo_name2class` diectionary defined in `offpcc/launch.py`. Within that algorithm folder, create a folder with the run_id, e.g., `1`. Simply put the saved actor (also actor summarizer for recurrent algorithms) into that inner most foler - they can be downloaded from the wandb website after your run finishes. Finally, go back into `offpcc`, and call

```pytho
python launch.py --env pendulum-v0 --algo sac --config configs/test/template_short.gin --run_id 1 --render
```

For `bumps-normal-test-v0`, you need to modify the `test_for_one_episode` function within `offpcc/basics/run_fns.py` because, for Pybullet environments, the `env.step` must only appear once before the `env.reset()` call.

