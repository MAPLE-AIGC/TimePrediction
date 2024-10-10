#! /bin/bash
cd /litiancheng/t2i
PROC_PER_NODES=1

# 使用 debugpy 调试模式
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py \
--nproc_per_node=$PROC_PER_NODES \
--rdzv_id=LD \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
--config configs/sd1-4.yaml
