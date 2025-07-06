ulimit -n 12288

for seed in 45 46 47 48 49
do
    python -u  main.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --train_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_train.json \
        --test_dataset /mnt/data1/mshee/ContrastiveFineTune/lavis-main-5/datasets/fhm_classification_test.json \
        --img_dir /mnt/data1/datasets/memes/fhm_finegrained/images/img \
        --seed $seed \
        --metric cos \
        --train \
        --evaluate \
        --gradient_checkpointing
done