# train
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 /opt/data/private/zyc/ML/transformer/main.py \
    --mode 'train' \
    --data_path '/opt/data/private/zyc/ML/transformer/dataset' \
    --d_model 512 \
    --n_head 8 \
    --n_encoder 6 \
    --n_decoder 6 \
    --dropout 0.1 \
    --epochs 10 \
    --output_size 96 \
    --input_size 96 \
    --lr 1e-3 \
    --batch_size 64 \
    --print_every 10 \
    --log_file '/opt/data/private/zyc/ML/transformer/logs/train.log' \
    --save_path '/opt/data/private/zyc/ML/transformer/ckpts/96' \
    --special_tokens \