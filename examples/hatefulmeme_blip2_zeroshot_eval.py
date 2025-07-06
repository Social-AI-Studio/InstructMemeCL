import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import argparse

model_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/gaozihan04/hatediffusion/LAVIS-main/model/blip2-opt-6.7b"
processor = Blip2Processor.from_pretrained(model_path)
half_precision = False
if half_precision:
    model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
else:
    model = Blip2ForConditionalGeneration.from_pretrained(model_path)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
model.to(device)

def blip2(image=None, prompt=None, use_prompt=False):
    if not use_prompt:
        inputs = processor(image, return_tensors="pt").to(device)
    else:
        if half_precision:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        else:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs)
    #exp1 # , do_sample=True, top_p=0.9, temperature=1, num_beams=1, max_length=30, min_length=1
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


# parse = argparse.ArgumentParser('BLIP2 - image captioning & visual question answering.')
# parse.add_argument('--image', type=str, default="examples/cats.jpeg", required=True, help='input image path.')
# parse.add_argument('--prompt', type=str, required=False, default="is this meme bad or good ? explain why .", help='prompt = question.')

import json

if __name__ == "__main__":
    # conda_env: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/weifei06/code/LAVIS/conda_env_lavis
    # args = parse.parse_args()
    # image = Image.open(args.image)
    # prompt = args.prompt # 模板:"Question: {} Answer:", e.g. "Question: how many cats are there? Answer:"
    # print(prompt)
    # if prompt is None:
    #     use_prompt = False
    # else:
    #     use_prompt = True
    data = json.load(open('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/memes/data/domain_splits/mem_test.json', 'rb'))
    datasize = len(data)
    print("datasize:   ", datasize)
    invalid_count = 0
    correct_count = 0
    for k, row in enumerate(data):
        label = row['label']
        img = row['img']
        text = row['clean_sent']
        img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/memes/mem/img_clean/" + img
        image = Image.open(img_path)
        prompt =  "This is an image with: \' { " + text + " } \' written on it. Is it hateful? Answer:"
        if k==0:
            print(prompt)
        generated_text = blip2(image, prompt, True)
        # print(generated_text)
        row['generation'] = generated_text
        if generated_text[:3].lower() == "yes":
            pred_label = 1
            if pred_label == label:
                correct_count += 1
        elif generated_text[:2].lower() == "no":
            pred_label = 0
            if pred_label == label:
                correct_count += 1
        else:
            print("invalid generation:   ", "img:    ", img, "generation:    ", generated_text, "label:  ", label)
            pred_label = -1
            invalid_count += 1
        row["pred_label"] = pred_label
        
        with open('examples/output/hateful_meme_blip2.json', 'a', encoding='utf-8') as f:
            json.dump(row, f, ensure_ascii=False)
            f.write('\n')
    print("correct_count:    ", correct_count, "invalid_count:   ", invalid_count)
    print("accuracy:    ", correct_count / (datasize - invalid_count))
    print("accuracy:    ", correct_count / (datasize))
    
        