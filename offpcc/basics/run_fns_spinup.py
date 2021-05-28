import torch.nn as nn
import gin

from spinup import ddpg_pytorch as ddpg
from spinup import td3_pytorch as td3
from spinup import sac_pytorch as sac


@gin.configurable(module=__name__)
def train_ddpg(
        env_fn,
        seed,
        hidden_dimensions=gin.REQUIRED,
        num_steps_per_epoch=gin.REQUIRED,
        num_epochs=gin.REQUIRED,
        capacity=gin.REQUIRED,
        gamma=gin.REQUIRED,
        polyak=gin.REQUIRED,
        lr=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        update_after=gin.REQUIRED,
        update_every=gin.REQUIRED,
        action_noise=gin.REQUIRED,
        num_test_episodes_per_epoch=gin.REQUIRED
):
    ddpg(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=list(hidden_dimensions), activation=nn.ReLU),
        seed=seed,
        steps_per_epoch=num_steps_per_epoch,
        epochs=num_epochs,
        replay_size=capacity,
        gamma=gamma,
        polyak=polyak,
        pi_lr=lr,
        q_lr=lr,
        batch_size=batch_size,
        start_steps=update_after,
        update_after=update_after,
        update_every=update_every,
        act_noise=action_noise,
        num_test_episodes=num_test_episodes_per_epoch,
        max_ep_len=env_fn().spec.max_episode_steps,
        logger_kwargs=dict(),  # automatically uploaded to wandb
        save_freq=50  # we don't actually care about saved models from spinup
    )


@gin.configurable(module=__name__)
def train_td3(
        env_fn,
        seed,
        hidden_dimensions=gin.REQUIRED,
        num_steps_per_epoch=gin.REQUIRED,
        num_epochs=gin.REQUIRED,
        capacity=gin.REQUIRED,
        gamma=gin.REQUIRED,
        polyak=gin.REQUIRED,
        lr=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        update_after=gin.REQUIRED,
        update_every=gin.REQUIRED,
        action_noise=gin.REQUIRED,
        target_noise=gin.REQUIRED,
        noise_clip=gin.REQUIRED,
        policy_delay=gin.REQUIRED,
        num_test_episodes_per_epoch=gin.REQUIRED
):
    td3(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=list(hidden_dimensions), activation=nn.ReLU),
        seed=seed,
        steps_per_epoch=num_steps_per_epoch,
        epochs=num_epochs,
        replay_size=capacity,
        gamma=gamma,
        polyak=polyak,
        pi_lr=lr,
        q_lr=lr,
        batch_size=batch_size,
        start_steps=update_after,
        update_after=update_after,
        update_every=update_every,
        act_noise=action_noise,
        target_noise=target_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        num_test_episodes=num_test_episodes_per_epoch,
        max_ep_len=env_fn().spec.max_episode_steps,
        logger_kwargs=dict(),
        save_freq=50
    )


@gin.configurable(module=__name__)
def train_sac(
        env_fn,
        seed,
        hidden_dimensions=gin.REQUIRED,
        num_steps_per_epoch=gin.REQUIRED,
        num_epochs=gin.REQUIRED,
        capacity=gin.REQUIRED,
        gamma=gin.REQUIRED,
        polyak=gin.REQUIRED,
        lr=gin.REQUIRED,
        alpha=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        update_after=gin.REQUIRED,
        update_every=gin.REQUIRED,
        num_test_episodes_per_epoch=gin.REQUIRED
):
    sac(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=list(hidden_dimensions), activation=nn.ReLU),
        seed=seed,
        steps_per_epoch=num_steps_per_epoch,
        epochs=num_epochs,
        replay_size=capacity,
        gamma=gamma,
        polyak=polyak,
        lr=lr,
        alpha=alpha,
        batch_size=batch_size,
        start_steps=update_after,
        update_after=update_after,
        update_every=update_every,
        num_test_episodes=num_test_episodes_per_epoch,
        max_ep_len=env_fn().spec.max_episode_steps,
        logger_kwargs=dict(),
        save_freq=50
    )