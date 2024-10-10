cd /huangzemin/lute/LLaMADiffusion/LLaMADiffusion
# export TORCH_DISTRIBUTED_DEBUG=INFO
~/envs/fetchA/bin/python train.py \
--config configs/test.yaml
# --config configs/test.yaml
# --config configs/qwen_pretrain_flow_opt.yaml

# old
# 108.75, 105.22, 106.79
# optimized
# 