# valid
export CUDA_VISIBLE_DEVICES=0

python /opt/data/private/zyc/ML/transformer/main.py \
    --mode 'dev' \
    --gpu 0 \
    --data_path '/opt/data/private/zyc/ML/transformer/dataset' \
    --output_size 96 \
    --input_size 96 \
    --d_model 512 \
    --n_head 8 \
    --n_encoder 6 \
    --n_decoder 6 \
    --dropout 0.1 \
    --batch_size 64 \
    --print_every 20 \
    --log_file '/opt/data/private/zyc/ML/transformer/logs/valid.log' \
    --save_path '/opt/data/private/zyc/ML/transformer/ckpts/' 