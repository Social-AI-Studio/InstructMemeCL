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
device = "cpu"
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
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


parse = argparse.ArgumentParser('BLIP2 - image captioning & visual question answering.')
parse.add_argument('--image', type=str, default="examples/cats.jpeg", required=True, help='input image path.')
parse.add_argument('--text', type=str, required=True, default="", help='TEXT=OCR TEXT')

if __name__ == "__main__":
    # conda_env: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/weifei06/code/LAVIS/conda_env_lavis
    args = parse.parse_args()
    image = Image.open(args.image)
    print(args.image)
    # print(args.text)
    use_prompt = True

    # prompt = "is it a hateful meme ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    prompt =  "This is an image with: \' { " + args.text + " } \' written on it. Is it hateful? Answer:"
    # prompt = "If you add the text \' " + args.text + " \' to this image to create a meme, is this new meme good or bad ? Please explain why ."
    print(prompt)
    generated_text = blip2(image, prompt, use_prompt)
    print(generated_text)

    # prompt = "is it a hateful meme ? explain the reason. "
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "is it a hateful meme ? explain why. "
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)


    # prompt = args.prompt # 模板:"Question: {} Answer:", e.g. "Question: how many cats are there? Answer:"
    # prompt = "If you add the text \' " + args.text + " \' to this image to create a meme, is this new meme good or bad ? Please explain the reason ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "If you add the text \' " + args.text + " \' to this image to create a meme, is this new meme good or bad ? Please explain why ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "If you add the text \' " + args.text + " \' to this image to create a meme, why is this new meme good or bad ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)



    # prompt = "Is this meme bad or good ? Please explain the reason ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Is this meme bad or good ? Explain the reason ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Is this meme bad or good ? Please explain why ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Is this meme bad or good ? Explain why ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Why is this meme bad or good ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Extract the text in the meme. Is this meme bad or good ? Explain the reason ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Extract the text in the meme. Is this meme bad or good ? Explain why ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Extract the text in the meme. Why is this meme bad or good ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "Extract the text in the meme. Why is this meme good or bad ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)


    # prompt = "can you extract the text in the meme ? is this meme good or bad ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "please extract the text in the meme , is this meme good or bad ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "please extract the text in the meme"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "is this meme good or bad ? explain why ."
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)

    # prompt = "why this meme is good or bad ?"
    # print(prompt)
    # generated_text = blip2(image, prompt, use_prompt)
    # print(generated_text)
