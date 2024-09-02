#! /bin/bash

NUM_GPUS_PER_WORKER=2
MASTER_PORT=29501

train_options=" \
       --savepath blip_uni_cross_mul \
       --batch_size 32 \
       --accumulation_steps 4 \
       --epochs 10 \
       --distributed True \
       --gpu_num ${NUM_GPUS_PER_WORKER} \
       --gpu_id '1,7' \
       --clear_visualizer \
       --fix_rate 0.7 \
       --lr 5e-04 \
       --lr-decay-style cosine \
       --warmup 0.0 \
       --rank_pair \
       --load_pair_store \
       --std_log \
       --valid_per_epoch 4 \
"

run_cmd="torchrun
        --nnodes=1
        --nproc_per_node=${NUM_GPUS_PER_WORKER}
        --master_port=${MASTER_PORT}
        main.py ${train_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
