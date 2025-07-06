"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.meme_datasets import MemeDataset, MemeEvalDataset, MemeDatasetTriplet


@registry.register_builder("meme_classification")
class MemeClassificationBuilder(BaseDatasetBuilder):
    train_dataset_cls = MemeDataset
    eval_dataset_cls = MemeEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/meme/defaults.yaml",
    }


@registry.register_builder("meme_classification_triplet")
class MemeClassificationBuilder(BaseDatasetBuilder):
    train_dataset_cls = MemeDatasetTriplet
    eval_dataset_cls = MemeEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/meme/defaults_triplet.yaml",
    }