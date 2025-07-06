from lavis.models import load_model_and_preprocess
import argparse


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


if __name__ == "__main__":
    device = "cuda"

    job_id = now()

    args = parse_args()
    cfg = Config(args)
    # print(cfg.__dict__)
    # print(cfg.config.run.output_dir)
    # exit()

    # init_distributed_mode(cfg.run_cfg) #是否用distributed mode

    setup_seeds(cfg)
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    
    # # Load the best checkpoint from config's output directory
    # config_output_dir = cfg.run_cfg.output_dir
    # checkpoint_path = os.path.join(config_output_dir, "checkpoint_best.pth")
    
    # if os.path.isfile(checkpoint_path):
    #     print(f"Loading best checkpoint from {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path)

    #     # Try to load state dict with strict=True first
    #     model.load_state_dict(checkpoint["model"])
    #     print("Checkpoint loaded successfully")
    # else:
    #     print(f"No checkpoint found at {checkpoint_path}")
    #     print("Using model without loading weights")

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    # runner.train()
    
    # Define output folder and check if files already exist
    output_folder = cfg.run_cfg.output_dir
    if output_folder is None:
        output_folder = os.path.join(runner.output_dir, "tsne_data")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    features_file = os.path.join(output_folder, "tsne_features.npy")
    targets_file = os.path.join(output_folder, "tsne_targets.npy")
    
    # Only compute t-SNE if the required files don't exist
    if not (os.path.exists(features_file) and os.path.exists(targets_file)):
        print(f"Computing t-SNE representations and saving to {output_folder}")
        runner.compute_and_save_tsne(num=20, output_folder=output_folder)
    else:
        print(f"t-SNE data already exists in {output_folder}, skipping computation")
    
    # Generate the plot using the existing or newly computed data
    plot_title = "t-SNE w anti-similarity contrastive tuning"
    plot_path = "tsne-ct-20.png"
    
    runner.plot_tsne(data_folder=output_folder, title=plot_title, savepath=plot_path)
    
    # # Original code commented out:
    # # runner.tsne(num=20, title="t-SNE w anti-similarity contrastive tuning", savepath="tsne-ct-20.png")
    # # runner.tsne(num=20, title="t-SNE wo anti-similarity contrastive tuning", savepath="tsne-ft-20.png")
    # # print("xxx")
    # # runner.tsne(num=100, title="t-SNE w anti-similarity contrastive tuning", savepath="tsne-ct.png")