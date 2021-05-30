import wandb

ORIGINAL = "pendulum-var-len-pvl-v0 ddpg mdp_ddpg.gin (sb3)"
NEW = "pendulum-var-len-pvl-v0 ddpg pvl_ddpg.gin (sb3)"

api = wandb.Api()
runs = api.runs('pomdpr/workshop')
for run in runs:
    if run.group == ORIGINAL:
        run.group = NEW
        run.update()
