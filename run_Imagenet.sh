data_root="/your/data/root"  # please set your data root path here


#########################
# Imagenet Resnet50
#########################

# Pretrained model after Sparse learning; please download it and place it in the path "sl-resume" first
# Or you can use the original Resnet50 of Pytorch and conduct sparse learning to achieve it
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 \
    --use_env main_imagenet_EP_resnet.py --data-path "$data_root"  \
    --model resnet50 --global-pruning --pretrained --target-flops 2.04  --reg 1e-4 \
    --print-freq 200 --workers 16 --equal_pruning  --prune --resnet_skip_block_last --train_naive_pruning \
    --lr-step-size 30 --batch-size 256 --epochs 100 --output-dir resnet50 \
    --sparsity_learning --sl-lr 0.08 --sl-lr-step-size 10  --sl-epochs 30 \
    --lr 0.04 --method Jacobian --amp --N_batchs 50 --wd_div 0 --lr_div 4 --group_reduction sum \
    --lr-scheduler cosineannealinglr  \
    --distill --coeff-ce 0.5 --coeff-label 0.5 --T 4 

#########################
# Imagenet MobileNetV2
#########################
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 18122 \
    --use_env main_imagenet_EP_mobilenet.py --data-path "$data_root"  \
    --model mobilenet_v2 --print-freq 200 \
    --method Jacobian --global-pruning --pretrained --amp --group_reduction sum --normalizer None  \
    --target-flops 0.154 --prune  --Jacobian_batch_size 256 --N_batchs 50 \
    --output-dir mobilenet_v2  \
    --mobile_ignore_last --isomorphic \
    --sparsity_learning --sl-lr 0.036 --sl-lr-step-size 1  --sl-epochs 150 \
    --train_naive_pruning --batch-size 512 --epochs 300 --lr 0.036 --lr-scheduler cosineannealinglr --wd  2e-5 


#########################
# Imagenet ViT
#########################
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 18101 \
    --use_env main_imagenet_EP_VIT.py --data-path "$data_root" \
    --model vit_b_16 --pretrained \
    --method Jacobian --train_naive_pruning --prune --global-pruning --target-flops 10 --model-ema --amp \
    --mixup-alpha 0.8 --auto-augment ra  --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --random-erase 0.25 \
    --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 \
    --opt adamw  --lr-scheduler cosineannealinglr --Jacobian_batch_size 64 --batch-size 128 --distill    --lr 0.000125 --wd 0.05 --epochs 100 --label-smoothing 0.11 --N_batchs 50 --equal_pruning --wd_div 0 --lr_div 5 \
    --output-dir vit
