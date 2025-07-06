ulimit -n 12288

python -u  main.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --train_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_train.json \
    --test_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_test.json \
    --img_dir /mnt/data1/datasets/memes/fhm_finegrained/images/img \
    --seed 40 \
    --warmup_steps 0 \
    --use_contrastive_loss \
    --margin 1.25 \
    --scale 1.0 \
    --metric cos \
    --train \
    --evaluate

for seed in 41 42 43 43
do
    # vanillia 
    python -u  main.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --train_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_train.json \
        --test_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_test.json \
        --img_dir /mnt/data1/datasets/memes/fhm_finegrained/images/img \
        --seed $seed \
        --warmup_steps 0 \
        --metric cos \
        --train \
        --evaluate


    # contrastive
    python -u  main.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --train_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_train.json \
        --test_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_test.json \
        --img_dir /mnt/data1/datasets/memes/fhm_finegrained/images/img \
        --seed $seed \
        --warmup_steps 0 \
        --use_contrastive_loss \
        --margin 1.5 \
        --scale 1.0 \
        --metric cos \
        --train \
        --evaluate

done