"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from sklearn.metrics import roc_auc_score


@registry.register_task("meme_classification")
class MemeClassificationTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        samples_p = {
            "image": samples["image"],
            "prompt": samples["text_input"],
        }
        # if samples["image_file"] 
        # if samples["image_file"] == ['86417.png', '92567.png', '63921.png', '72864.png', '30586.png', '83264.png', '93041.png', '97305.png']:
        #     print("xxxx")
        #     print("yyyy")
        output = model.generate(
            samples_p,
            use_nucleus_sampling=True,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            roc_auc_compute=True, # True, mannually added by gzh for roc_auc compute, output vanilla output logits
            output_scores=True, # True, added for scores return
            return_dict_in_generate=True, # True, added for scores return
        )

       
        img_file = samples["image_file"]
        text_input = samples["text_input"]
        label = samples["label"].tolist()
        output_score = output[1].tolist()
        for output_text, output_score, text_input, label, img_file in zip(output[0], output_score, text_input, label, img_file):
            results.append({"image_file": img_file, "text_input":text_input, "label": label, "output_text": output_text, "score": output_score})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
    
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_file",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                val_result=val_result , eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, val_result, eval_result_file, split_name):
        
        scores = []
        preds = []
        labels = []
        log_stats = {}
        hate = 0
        nothate = 0
        for value in val_result:
            if value["output_text"][0:3].lower() == "yes":
                pred = 1
            elif value["output_text"][0:2].lower() == "no":
                pred = 0
            else:
                pred = -1
                print("invalid output:   ", value["output_text"])
            preds.append(pred)
            labels.append(value["label"])
            scores.append(value['score'])

        log_stats['auc'] = roc_auc_score(labels, scores)
        log_stats['acc'] = sum(1 for x, y in zip(preds, labels) if x == y)/len(labels)
        log_stats['agg_metrics'] = log_stats['auc'] + log_stats['acc']
        print(log_stats)


        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        return log_stats

