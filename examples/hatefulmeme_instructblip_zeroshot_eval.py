from lavis.models import load_model_and_preprocess
import argparse
from PIL import Image
import json

# parse = argparse.ArgumentParser('BLIP2 - image captioning & visual question answering.')
# parse.add_argument('--image', type=str, default="examples/cats.jpeg", required=True, help='input image path.')
# parse.add_argument('--text', type=str, required=True, default="", help='TEXT=OCR TEXT')

if __name__ == "__main__":
    # conda_env: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/weifei06/code/LAVIS/conda_env_lavis
    # args = parse.parse_args()
    # image = Image.open(args.image)
    # print(args.image)
    device = "cuda"


    # loads InstructBLIP model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    # prepare the image

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
        try:
            image = vis_processors["eval"](image).unsqueeze(0).to(device)
        except:
            print("error:  ", img_path)
            continue
        generated_text = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1, num_beams=1, max_length=30, min_length=1)
        if k==0:
            print(prompt)
        # print(generated_text)
        row['generation'] = generated_text
        if generated_text[0][:3].lower() == "yes":
            pred_label = 1
            if pred_label == label:
                correct_count += 1
        elif generated_text[0][:2].lower() == "no":
            pred_label = 0
            if pred_label == label:
                correct_count += 1
        else:
            print("invalid generation:   ", "img:    ", img, "generation:    ", generated_text, "label:  ", label)
            pred_label = -1
            invalid_count += 1
        row["pred_label"] = pred_label
        
        with open('output/fhm_blip2_vicuna7b_instruct_prompt1.json', 'a', encoding='utf-8') as f:
            json.dump(row, f, ensure_ascii=False)
            f.write('\n')
    print("correct_count:    ", correct_count, "invalid_count:   ", invalid_count)
    print("remove invalid: accuracy:    ", correct_count / (datasize - invalid_count))
    print("invalid is false: accuracy:    ", correct_count / (datasize))

    # # -------------------different prompt--------------------------

    # invalid_count = 0
    # correct_count = 0
    # for k, row in enumerate(data):
    #     label = row['label']
    #     img = row['img']
    #     text = row['clean_sent']
    #     img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/memes/mem/img_clean/" + img
    #     image = Image.open(img_path)

    #     prompt =  "If you add the text \' { " + text + " } \' to this image to create a meme, is this new meme good or bad ? Please explain why ."
    #     try:
    #         image = vis_processors["eval"](image).unsqueeze(0).to(device)
    #     except:
    #         print("error:  ", img_path)
    #         continue
    #     generated_text = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1, num_beams=1, max_length=30, min_length=1)
    #     if k==0:
    #         print(prompt)
    #     # print(generated_text)
    #     row['generation'] = generated_text
    #     if generated_text[0][:3].lower() == "yes":
    #         pred_label = 1
    #         if pred_label == label:
    #             correct_count += 1
    #     elif generated_text[0][:2].lower() == "no":
    #         pred_label = 0
    #         if pred_label == label:
    #             correct_count += 1
    #     else:
    #         print("invalid generation:   ", "img:    ", img, "generation:    ", generated_text, "label:  ", label)
    #         pred_label = -1
    #         invalid_count += 1
    #     row["pred_label"] = pred_label
        
    #     with open('output/fhm_blip2_vicuna7b_instruct_prompt2.json', 'a', encoding='utf-8') as f:
    #         json.dump(row, f, ensure_ascii=False)
    #         f.write('\n')
    # print("correct_count:    ", correct_count, "invalid_count:   ", invalid_count)
    # print("remove invalid: accuracy:    ", correct_count / (datasize - invalid_count))
    # print("invalid is false: accuracy:    ", correct_count / (datasize))


    # #-----------------------------------diffenrent prompt--------------------

    # invalid_count = 0
    # correct_count = 0
    # for k, row in enumerate(data):
    #     label = row['label']
    #     img = row['img']
    #     text = row['clean_sent']
    #     img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/memes/mem/img_clean/" + img
    #     image = Image.open(img_path)

    #     prompt =  "This is an image with: \' " + text + " \' written on it. Is it hateful? Answer:"
    #     try:
    #         image = vis_processors["eval"](image).unsqueeze(0).to(device)
    #     except:
    #         print("error:  ", img_path)
    #         continue
    #     generated_text = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1, num_beams=1, max_length=30, min_length=1)
    #     if k==0:
    #         print(prompt)
    #     # print(generated_text)
    #     row['generation'] = generated_text
    #     if generated_text[0][:3].lower() == "yes":
    #         pred_label = 1
    #         if pred_label == label:
    #             correct_count += 1
    #     elif generated_text[0][:2].lower() == "no":
    #         pred_label = 0
    #         if pred_label == label:
    #             correct_count += 1
    #     else:
    #         print("invalid generation:   ", "img:    ", img, "generation:    ", generated_text, "label:  ", label)
    #         pred_label = -1
    #         invalid_count += 1
    #     row["pred_label"] = pred_label
        
    #     with open('output/fhm_blip2_vicuna7b_instruct_prompt3.json', 'a', encoding='utf-8') as f:
    #         json.dump(row, f, ensure_ascii=False)
    #         f.write('\n')
    # print("correct_count:    ", correct_count, "invalid_count:   ", invalid_count)
    # print("remove invalid: accuracy:    ", correct_count / (datasize - invalid_count))
    # print("invalid is false: accuracy:    ", correct_count / (datasize))

    # # -------------------different prompt--------------------------

    # invalid_count = 0
    # correct_count = 0
    # for k, row in enumerate(data):
    #     label = row['label']
    #     img = row['img']
    #     text = row['clean_sent']
    #     img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-automl/common/memes/mem/img_clean/" + img
    #     image = Image.open(img_path)

    #     prompt =  "If you add the text \' " + text + " \' to this image to create a meme, is this new meme good or bad ? Please explain why ."
    #     try:
    #         image = vis_processors["eval"](image).unsqueeze(0).to(device)
    #     except:
    #         print("error:  ", img_path)
    #         continue
    #     generated_text = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1, num_beams=1, max_length=30, min_length=1)
    #     if k==0:
    #         print(prompt)
    #     # print(generated_text)
    #     row['generation'] = generated_text
    #     if generated_text[0][:3].lower() == "yes":
    #         pred_label = 1
    #         if pred_label == label:
    #             correct_count += 1
    #     elif generated_text[0][:2].lower() == "no":
    #         pred_label = 0
    #         if pred_label == label:
    #             correct_count += 1
    #     else:
    #         print("invalid generation:   ", "img:    ", img, "generation:    ", generated_text, "label:  ", label)
    #         pred_label = -1
    #         invalid_count += 1
    #     row["pred_label"] = pred_label
        
    #     with open('output/fhm_blip2_vicuna7b_instruct_prompt4.json', 'a', encoding='utf-8') as f:
    #         json.dump(row, f, ensure_ascii=False)
    #         f.write('\n')
    # print("correct_count:    ", correct_count, "invalid_count:   ", invalid_count)
    # print("remove invalid: accuracy:    ", correct_count / (datasize - invalid_count))
    # print("invalid is false: accuracy:    ", correct_count / (datasize))

    # # ['yes', 'true', 'no', 'false']

    