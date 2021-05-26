import wandb


def log_current_row_to_wandb(current_row: dict):
    wandb.log({
        'epoch': current_row['Epoch'],
        'timestep': current_row['TotalEnvInteracts'] + 1,
        'train_ep_len': current_row['EpLen'],
        'train_ep_ret': current_row['AverageEpRet'],
        'test_ep_len': current_row['TestEpLen'],
        'test_ep_ret': current_row['AverageTestEpRet'],
    })
