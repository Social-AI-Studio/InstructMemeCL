from PIL import Image
import json
import json
import os


data_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/memes/mem/img/"

hateful_path = "ct_hateful.json"
nonhateful_path = "ct_nonhateful.json"

hate = json.load(open(hateful_path,'rb'))
nonhate = json.load(open(nonhateful_path,'rb'))


hateimages = [Image.open(data_path + img_path) for img_path in hate]
nonhateimages = [Image.open(data_path + img_path) for img_path in nonhate]

hate_savepath = "z-ct_hateful"
nonhate_savepath = "z-ct_nonhateful"
for i in range(len(hate)):
    save_path = hate_savepath + '/' + hate[i]
    image = hateimages[i]
    image.save(save_path)

for i in range(len(nonhate)):
    save_path = nonhate_savepath + '/' + nonhate[i]
    image = nonhateimages[i]
    image.save(save_path)