cd /huangzemin/lute/LLaMADiffusion/LLaMADiffusion

PROC_PER_NODES=4

echo "RDZV backend: $MASTER_ADDR:$MASTER_PORT"

/root/envs/fetchA/bin/torchrun \
--standalone \
--nproc-per-node=$PROC_PER_NODES \
--rdzv-id=LD \
--rdzv-backend=c10d \
--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
train.py  \
--config configs/qwen_pretrain_flow_4proc.yaml