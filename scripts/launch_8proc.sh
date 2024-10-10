#! /bin/bash
cd /litiancheng/t2i
PROC_PER_NODES=1


torchrun \
--standalone \
--nproc-per-node=$PROC_PER_NODES \
--rdzv-id=LD \
--rdzv-backend=c10d \
--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
train.py  \
--config configs/sd1-4.yaml