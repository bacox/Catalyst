import logging
from typing import Tuple, Union
import wandb
from wandb import AlertLevel, sdk as wandb_sdk
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)


def init_reporting(exp_name: str, name: str, meta_data:dict = {}) -> wandb_sdk.wandb_run.Run:
    logging.info(f'Reporting with metadata: {meta_data}')
    print(f'Reporting [{name}] with metadata: {meta_data}')
    return wandb.init(
        # set the wandb project where this run will be logged
        project=exp_name,
        name= name,
        # track hyperparameters and run metadata
        config=meta_data,
        # settings=wandb.Settings(console="off")
    ) # type: ignore

def report_data(wandb_obj: Union[wandb_sdk.wandb_run.Run, None], data: dict):
    if wandb_obj:
        wandb_obj.log(data)

def finish_exp(wandb_obj: wandb_sdk.wandb_run.Run):
    if wandb_obj:
        wandb_obj.finish(quiet=True)
    else:
        logging.warning('No wandb object to finish')

def finish_reporting(wandb_obj: wandb_sdk.wandb_run.Run, exp_name: str, message: str):
    
    logging.info(message)
    if wandb_obj:
        wandb_obj.alert(
            title=f'Experiment Finished "{exp_name}"',
            text=message,
            level=AlertLevel.INFO,
            wait_duration=0
            # wait_duration=timedelta(minutes=5)
        )