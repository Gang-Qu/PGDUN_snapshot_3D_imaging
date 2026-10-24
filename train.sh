# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,8,9 python -m torch.distributed.launch --master_port=20335 --nproc_per_node=8 train.py --distributed=True

# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --master_port=20335 --nproc_per_node=4 train.py --distributed=True


cd /qugang/DH_PSF_0820
source activate
conda activate mamba2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=20335 --nproc_per_node=4 train.py --distributed=True
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --distributed=True
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --distributed=True
# python test.py
