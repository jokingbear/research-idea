import wandb
import time
import numpy as np

from tqdm import tqdm


wandb.init(job_type='training', dir='check', project='test wandb')

for i in tqdm(range(50)):
    wandb.log({"epoch": i, "loss": np.random.normal()})
    time.sleep(1)
