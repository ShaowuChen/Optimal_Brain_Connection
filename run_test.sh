CUDA_VISIBLE_DEVICES=0 python main_cifar.py --model resnet56 --dataset cifar10 --speed-up 1.1 --N-batchs 10 --save-suffix 2 --restore /data/csw/optimal_Brain_Pruning2/Optimal_Brain_Pruning/results/base_model/DepGraph/cifar10_resnet56.pth --finetune --method Jacobian --global-pruning --equivalent --N-batchs 50 --mode prune --lr-div 0.5 --wd-div 0 --select CB --normalizer None --group-reduction sum --normalizer_for_sl max --output-dir resnet56 --sparsity-learning  --sl-total-epochs 1 --total-epochs 1



CUDA_VISIBLE_DEVICES=3 python main_cifar.py --model vgg19 --dataset cifar100 --speed-up 1.1 --save-suffix 2 --restore /data/csw/optimal_Brain_Pruning2/Optimal_Brain_Pruning/results/base_model/DepGraph/cifar100_vgg19.pth --finetune --method Jacobian --equivalent --N-batchs 50 --mode prune --lr-div 5 --wd-div 1 --select CB --normalizer None --group-reduction first --normalizer_for_sl max --output-dir vgg --sparsity-learning --sl-total-epochs 1 --total-epochs 1


data_root="/data/csw/dataset/ILSVRC2012"  # please set your data root path here
CUDA_VISIBLE_DEVICES=0  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 18119 \
    --use_env main_imagenet_EP_resnet.py --data-path "$data_root"  \
    --model resnet50 --global-pruning --pretrained --target-flops 4  --reg 1e-4 \
    --print-freq 200 --workers 16 --equal_pruning  --prune --resnet_skip_block_last \
    --lr-step-size 30 --batch-size 8 --epochs 1 --output-dir resnet50 \
    --sparsity_learning --sl-lr 0.08 --sl-lr-step-size 10  --sl-epochs 1 \
    --lr 0.04 --method Jacobian --amp --N_batchs 1 --wd_div 0 --lr_div 4 --group_reduction sum \
    --lr-scheduler cosineannealinglr  \
    --distill --coeff-ce 0.5 --coeff-label 0.5 --T 4 --Jacobian_batch_size 8

data_root="/data/csw/dataset/ILSVRC2012"
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 18122 \
    --use_env main_imagenet_EP_mobilenet.py --data-path "$data_root"  \
    --model mobilenet_v2 --print-freq 200 \
    --method Jacobian --global-pruning --pretrained --amp --group_reduction sum --normalizer None \
    --target-flops 0.3 --prune  --Jacobian_batch_size 8 --N_batchs 1 \
    --output-dir mobilenet_v2  \
    --mobile_ignore_last --isomorphic \
    --sparsity_learning --sl-lr 0.036 --sl-lr-step-size 1  --sl-epochs 1 \
    --train_naive_pruning --batch-size 8 --epochs 1 --lr 0.036 --lr-scheduler cosineannealinglr --wd  2e-5 


CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 18101 \
    --use_env main_imagenet_EP_VIT.py --data-path "$data_root" \
    --model vit_b_16 --pretrained \
    --method Jacobian --train_naive_pruning --prune --global-pruning --target-flops 17 --model-ema --amp \
    --mixup-alpha 0.8 --auto-augment ra  --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --random-erase 0.25 \
    --lr-warmup-method linear --lr-warmup-epochs 1 --lr-warmup-decay 0.033 \
    --opt adamw  --lr-scheduler cosineannealinglr --Jacobian_batch_size 4 --batch-size 4 --distill    --lr 0.000125 --wd 0.05 --epochs 2 --label-smoothing 0.11 --N_batchs 1 --equal_pruning --wd_div 0 --lr_div 5 \
    --output-dir vit --lr-step-size 1
    