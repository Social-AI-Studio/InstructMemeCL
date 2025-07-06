import os
import torch
import json
import argparse

from transformers import EarlyStoppingCallback

import random
import numpy as np
import torch.backends.cudnn as cudnn
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig

from trl import SFTTrainer, SFTConfig

from datasets import load_dataset

from PIL import Image

from functools import partial

from qwen_vl_utils import process_vision_info
from loss import ContrastiveLossSmoother

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO)


#### OLD

# def softmax(x):
#     e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#     return e_x / e_x.sum(axis=-1, keepdims=True)

# def compute_metrics(eval_predictions):
#     preds, labels, scores = [], [], []
#     log_stats = {"invalid_outputs": 0}

#     predictions = eval_predictions.predictions[0]
#     targets = eval_predictions.label_ids

#     predictions = predictions[..., :-1, :]
#     targets = targets[..., 1:]
    
#     proba_scores = softmax(predictions[:, :, [2753, 9454]])  # (bz, seq_len, vocab_size) -> (bz, seq_len)
#     proba_scores = proba_scores[:, :, 1]
    
#     predictions = np.argmax(predictions, axis=-1)  #  (bz, seq_len, vocab_size) -> (bz, seq_len)

#     mask = targets != -100
#     predictions = predictions[mask].reshape(mask.shape[0], -1)
#     proba_scores = proba_scores[mask].reshape(mask.shape[0], -1)
#     targets = targets[mask].reshape(mask.shape[0], -1)
    
#     for idx in range(predictions.shape[0]):
#         print(predictions[idx])

#         # For Qwen2.5VL the fourth last token is the token we are interested in
#         # 2753: No
#         # 9454: Yes
#         def get_label(token_id):
#             if token_id == 2753:
#                 return 0
#             elif token_id == 9454:
#                 return 1
#             else:
#                 return -1

#         pred = get_label(predictions[idx][0])
#         label = get_label(targets[idx][0])
#         assert label in [0, 1], f"Invalid label: {label}"
#         print(pred, label, proba_scores[idx][0])

#         preds.append(pred)
#         labels.append(label)
#         scores.append(proba_scores[idx][0])

#     log_stats["auc"] = roc_auc_score(labels, scores)
#     log_stats['acc'] = sum(1 for x, y in zip(preds, labels) if x == y)/len(labels)
#     log_stats['agg_metrics'] = log_stats['auc'] + log_stats['acc']

#     return log_stats


def preprocess_logits_for_metrics(predictions, labels):
    predictions = predictions[0][..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    # Retrieve the highest likelihood tokens at each step
    predictions_argmax = torch.argmax(predictions, dim=-1)  # (bz, seq_len, vocab_size) -> (bz, seq_len)

    # Mask out the starting input tokens (i.e., padding tokens)
    mask = labels != -100
    predictions_argmax = predictions_argmax[mask].reshape(mask.shape[0], -1)

    # Get the logits score for the two tokens
    indices = mask.nonzero(as_tuple=True)
    predictions_score = predictions[..., [2753, 9454]]  # (bz, seq_len, logits) -> (bz, seq_len, 2)
    predictions_score = predictions_score[indices[0], indices[1]].reshape(mask.shape[0], -1, 2)
    predictions_score = predictions_score[:, 0, :].squeeze() # Retrieve the first position

    # Do a softmax to convert into probabilites
    predictions_score = F.softmax(predictions_score, dim=-1)

    return torch.stack([predictions_argmax[:, 0], predictions_score[:, 1]], dim=-1)

def compute_metrics(eval_predictions):
    predictions = eval_predictions.predictions
    targets = eval_predictions.label_ids

    # Remove the first token 
    targets = targets[..., 1:]

    # Because they are from different batches, they will eventually be padded both left and right
    mask = targets != -100
    targets = targets[mask].reshape(mask.shape[0], -1)

    preds, labels, scores = [], [], []
    log_stats = {"invalid_outputs": 0}    
    for idx in range(predictions.shape[0]):

        # For Qwen2.5VL the fourth last token is the token we are interested in
        # 2753: No
        # 9454: Yes
        def get_label(token_id):
            if token_id == 2753:
                return 0
            elif token_id == 9454:
                return 1
            else:
                return -1

        pred = get_label(predictions[idx][0])
        label = get_label(targets[idx][0])

        assert label in [0, 1], f"Invalid label: {label}"

        preds.append(pred)
        labels.append(label)
        scores.append(predictions[idx][1])

    log_stats["auc"] = roc_auc_score(labels, scores)
    log_stats['acc'] = sum(1 for x, y in zip(preds, labels) if x == y)/len(labels)
    log_stats['agg_metrics'] = log_stats['auc'] + log_stats['acc']

    return log_stats


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def load_model(model_id):
    # Load model directly
    processor = AutoProcessor.from_pretrained(
        model_id
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        attn_implementation="flash_attention_2"
    )

    return model, processor

def convert_to_conversation(instruction: str, image: Image, label: str = False):
    conversation = [
        { "role": "user",
            "content" : [
                {"type" : "image", "image" : image},
                {"type" : "text",  "text"  : instruction}
            ]
        },
    ]
    
    if label:
        conversation.append(
            { "role": "assistant",
                "content" : [
                    {"type" : "text", "text" : label}
                ]
            }
        )

    return conversation

def load_dataset(filepath: str, img_dir: str, inference: bool):
    with open(filepath) as f:
        data = json.load(f)

    # Preprocess the dataset
    messages = []
    for d in data:
        # Load the image
        image_filepath = os.path.join(img_dir, d["image"])
        image = Image.open(image_filepath)

        # Convert to conversation
        if inference:
            m = convert_to_conversation(d["text_input"], image, d["answers"])
        else:
            m = convert_to_conversation(d["text_input"], image)

        messages.append(m)

    return messages

# Create a data collator to encode text and image pairs
def collate_fn(examples, processor):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    # labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # # Ignore the image token index in the loss computation (model specific)
    # # if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
    # #     image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    # # else:
    # image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # # Mask image token IDs in the labels
    # for image_token_id in image_tokens:
    #     labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    # all start_message <|im_start|> tokens, we keep only the last message so the only message that belongs to the assistant
    for cur_num, cur_local_coord in zip(*torch.where(labels == 151644)):
        labels[int(cur_num), :cur_local_coord + 3] = -100

    # print(labels.shape)
    batch["labels"] = labels  # Add labels to the batch
    # for idx in range(len(batch['input_ids'])):
    #     print(batch['input_ids'][idx].shape, batch['labels'][idx].shape)
    #     # debug = batch['labels'][idx, -10:]
    #     # debug = debug[(debug == -100)]
    #     print("input_ids:", batch['input_ids'][idx])
    #     print("input_ids:", processor.decode(batch['input_ids'][idx], skip_special_tokens=False).replace('\n', '[NL]'))
    #     print("input_ids:", batch['labels'][idx])
    #     print("input_ids:", processor.decode(batch['labels'][idx], skip_special_tokens=False).replace('\n', '[NL]'))
    #     # print("input_ids:", processor.decode(batch['input_ids'][idx, -10:], skip_special_tokens=False).replace('\n', '[NL]'))
    #     # print("labels:", processor.decode(debug, skip_special_tokens=False))
    #     exit()
    # print(batch['input_ids'][-10:])
    # print(batch['labels'][-10:])

    return batch  # Return the prepared batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to the testing dataset")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the testing dataset")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to load")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
   
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    
    parser.add_argument("--use_contrastive_loss", action="store_true", help="Whether to use ContrastiveSmoothLoss")
    parser.add_argument("--margin", type=float, default=-1)
    parser.add_argument("--scale", type=float, default=-1)
    parser.add_argument("--metric", type=str, choices=['cos', 'l2'], default=None)
    args = parser.parse_args()

    setup_seeds(args.seed)

    if args.use_contrastive_loss:
        assert args.margin > 0, "Margin must be greater than 0"
        assert args.scale > 0, "Scale must be greater than 0"
        assert args.metric is not None, "Metric must be provided"

    train_dataset = load_dataset(args.train_dataset, args.img_dir, True)
    test_dataset = load_dataset(args.test_dataset, args.img_dir, True)

    model, processor = load_model("Qwen/Qwen2.5-VL-7B-Instruct")

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=16,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    
    # Set padding_side to left (important for Flash Attention)
    processor.tokenizer.padding_side = 'left'

    # Configure training arguments
    project_name = "ContrastiveFT-Qwen2.5-7B"
    run_name = f"contrastive-seed{args.seed}-{args.lr}-{args.warmup_steps}-{args.metric}_{args.margin}_{args.scale}" if args.use_contrastive_loss else f"vanilla-seed{args.seed}-{args.lr}-{args.warmup_steps}" 
    output_dir = f"{project_name}/{args.metric}_{args.margin}_{args.scale}/{run_name}"

    training_args = SFTConfig(
        output_dir=output_dir,  # Directory to save the model
        logging_dir="./logs",
        save_total_limit=1,

        num_train_epochs=10,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        # eval_accumulation_steps=4, # offload predictions to CPU to avoid OOM
        gradient_accumulation_steps=4,  # Steps to accumulate gradients
        gradient_checkpointing=args.gradient_checkpointing,  # Enable gradient checkpointing for memory efficiency

        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=args.lr,  # Learning rate for training
        lr_scheduler_type="cosine",  # Type of learning rate scheduler

        # Logging and evaluation
        logging_steps=50,  # Steps interval for logging
        eval_strategy="epoch",  # Strategy for evaluation
        save_strategy="best",  # Strategy for saving the model
        # eval_steps=10,  # Steps interval for evaluation
        # save_steps=20,  # Steps interval for saving
        metric_for_best_model="eval_agg_metrics",  # Metric to evaluate the best model
        greater_is_better=True,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training

        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        warmup_steps=args.warmup_steps,  # Ratio of total steps for warmup

        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # report_to=None,  # Reporting tool for tracking metrics

        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing

        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        label_names=["labels"],  # Names of the label columns
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    import wandb

    wandb.init(
        project=project_name,  # change this
        name=run_name,  # change this
        config=training_args,
    )

    if args.train:
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=partial(collate_fn, processor=processor),
            peft_config=peft_config,
            tokenizer=processor.tokenizer,
            compute_loss_func=ContrastiveLossSmoother(args.metric, args.margin, args.scale) if args.use_contrastive_loss else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback]
        )

        trainer.train()
        trainer.save_model(training_args.output_dir)

    if args.evaluate:
        if not args.train:
            model.load_adapter(output_dir)
            
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=partial(collate_fn, processor=processor),
                tokenizer=processor.tokenizer,
                compute_loss_func=ContrastiveLossSmoother(args.metric, args.margin, args.scale) if args.use_contrastive_loss else None,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=compute_metrics
            )

        print(trainer.evaluate())

main()