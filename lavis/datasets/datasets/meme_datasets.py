"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import os
from collections import OrderedDict
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import pickle as pkl
import numpy as np
import random



class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "text_input": sample["text_input"],
                "image": sample["image"],
                "text_output": ann["answers"],
            }
        )
    
class MemeDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        return {
            "image": image,
            "text_input": text_input,
            "text_output": ann["answers"],
        }


class MemeEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])

        return {
            "image": image,
            "text_input": text_input,
            "text_output": ann["answers"],
            "image_id": ann["id"],
            "image_file": ann["image"],
            "label": ann["label"]
        }


def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

class MemeDatasetTriplet(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, config):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.use_preprocess_sampling=config.use_preprocess_sampling
        self.sampling_topk=config.sampling_topk
        clip_sim_path = config.clip_sim_path
        if self.use_preprocess_sampling:
            print ('Using demonstration sampling strategy...')
            print ('Sampling from top:',self.sampling_topk,'examples')
            print ('Clip feature similarity path:',clip_sim_path)
            self.clip_feature=load_pkl(clip_sim_path)
        
        self.support_examples=self.annotation
        self.prepare_exp()

    def prepare_exp(self):
        ###add sampling
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        
        for query_idx in range(len(self.annotation)):
            anchor_label = self.annotation[query_idx]["label"]
            positive_list = []
            negative_list = []
            if self.use_preprocess_sampling:
                #filter same demonstration
                candidates= [support_idx for support_idx in support_indices
                                if support_idx != query_idx]


                anchor_clip_sim=self.clip_feature[self.annotation[query_idx]['image']]

                img_sim = anchor_clip_sim["clean_img"]
                text_sim = anchor_clip_sim["text"]
                sim_score = np.maximum(img_sim, text_sim) #[8500,]
                sim_score = np.delete(sim_score, query_idx) #[8499,]
                
                desc_sorted_index = np.argsort(sim_score)[::-1]

                positive_num=0
                negative_num=0
                num_valid = self.sampling_topk

                
                for i in range(desc_sorted_index.shape[0]):
                    # print(desc_sorted_index[i])
                    support_idx = candidates[desc_sorted_index[i]]
                    cur_label = self.support_examples[support_idx]['label']
                    if cur_label == anchor_label and positive_num < num_valid:
                        positive_num += 1
                        positive_list.append(support_idx)

                    if cur_label != anchor_label and negative_num < num_valid:
                        negative_num += 1
                        negative_list.append(support_idx)

                    if positive_num==num_valid and positive_num==num_valid:
                        break
            else: 
                #exclude the current example during training
                for support_idx in support_indices:
                    if support_idx != query_idx:
                        if self.support_examples[support_idx]['label'] == anchor_label:
                            positive_list.append(support_idx)
                        else:
                            negative_list.append(support_idx)
                    else:
                        continue
            #available indexes for supporting examples
            self.example_idx.append((query_idx, positive_list, negative_list))


    def get_one_sample(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        text_input = self.text_processor(ann["text_input"])
        text_output = ann["answers"]
        return image, text_input, text_output
        
    def __getitem__(self, index):
        _, positive_list, negative_list = self.example_idx[index]
        positive_id = random.choice(positive_list)
        negative_id = random.choice(negative_list)

        image, text_input, text_output = self.get_one_sample(index)
        image_p, input_p, output_p = self.get_one_sample(positive_id)
        image_n, input_n, output_n = self.get_one_sample(negative_id)

        image_list = (image, image_p, image_n)
        text_list = (text_input, input_p, input_n)
        output_list=(text_output, output_p , output_n)

        return {
            "image": image_list,
            "text_input": text_list,
            "text_output": output_list,
        }
    
    
    
    