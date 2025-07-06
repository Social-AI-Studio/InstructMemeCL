from lavis.models import load_model_and_preprocess
import argparse
from PIL import Image
import json
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lavis.datasets.datasets.meme_dataset import MemeDatasetTriplet
from torch.utils.data import DataLoader


"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument('--meme',type=bool,default=True)
    

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg) #是否用distributed mode

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    device = "cuda"


    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg) #是否用distributed mode

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.tsne()

    # memedataset = MemeDataset(vis_processors, text_processor, vis_root, ann_paths)
    # loader = iter(DataLoader(
    #                     memedataset,
    #                     batch_size=16,
    #                     num_workers=8,
    #                     pin_memory=True,
    #                     shuffle=False
    #                 ))
    # count=0
    # while count<100:
    #     samples = next(loader)
    #     count += 1
    
    #     hiddenstates = model.forward_tsne(samples)
    #     datax=hiddenstates[4].cpu().numpy()
    #     tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=6.0)
    #     tsne_data=tsne.fit_transform(datax)
        
    #     show_pic(tsne_data, target, 't-SNE')

        