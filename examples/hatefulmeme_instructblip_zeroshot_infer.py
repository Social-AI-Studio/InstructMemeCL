# import torch
# import requests
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import argparse

# model_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/gaozihan04/hatediffusion/LAVIS-main/model/blip2-opt-6.7b"
# processor = Blip2Processor.from_pretrained(model_path)
# half_precision = False
# if half_precision:
#     model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
# else:
#     model = Blip2ForConditionalGeneration.from_pretrained(model_path)
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# model.to(device)



# def blip2(image=None, prompt=None, use_prompt=False):
#     if not use_prompt:
#         inputs = processor(image, return_tensors="pt").to(device)
#     else:
#         if half_precision:
#             inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
#         else:
#             inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
#     generated_ids = model.generate(**inputs)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     return generated_text

from lavis.models import load_model_and_preprocess
import argparse
from PIL import Image

parse = argparse.ArgumentParser('BLIP2 - image captioning & visual question answering.')
parse.add_argument('--image', type=str, default="examples/cats.jpeg", required=True, help='input image path.')
parse.add_argument('--text', type=str, required=True, default="", help='TEXT=OCR TEXT')

if __name__ == "__main__":
    # conda_env: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/weifei06/code/LAVIS/conda_env_lavis
    args = parse.parse_args()
    image = Image.open(args.image)
    print(args.image)
    device = "cuda"


    # loads InstructBLIP model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    # prepare the image
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    prompt =  "This is an image with: \' { " + args.text + " } \' written on it. Is it hateful? Answer:"
    # prompt = "If you add the text \' " + args.text + " \' to this image to create a meme, is this new meme good or bad ? Please explain why ."
    print(prompt)
    generated_text = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1, num_beams=1, max_length=30, min_length=1)
    print(generated_text)

    # ['yes', 'true', 'no', 'false']

    