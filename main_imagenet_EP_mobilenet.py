import datetime
import os, sys
import time
import warnings
import registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from engine.utils.imagenet_utils import presets, transforms, utils, sampler
import torch
import torch.utils.data
import torchvision
#from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import torch_pruning as tp 
from functools import partial
import torch.distributed as dist

# this one have to be imported after torch_pruning;
# otherwise ddp would be very slow; and I dont know why;
# numpy=1.26.4(conda install); 
# torch=2.5.1(pip install); 
# torch_pruning=1.5.1(pip install); 
import numpy as np  
from Selfmake_Importance import GroupJacobianImportance_accumulate
import copy
from typing import List, Optional, Tuple
import json, logging
import argparse

from torch.utils.tensorboard import SummaryWriter


def get_args_parser(add_help=True):
    

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/dev/shm/ILSVRC2012/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float, help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size" , default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=200, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    
    # pruning parameters
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--method", type=str, default='l1')
    parser.add_argument("--global-pruning", default=False, action="store_true")
    parser.add_argument("--target-flops", type=float, default=2.0, help="GFLOPs of pruned model")
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4)
    parser.add_argument("--max-pruning-ratio", default=1.0, type=float, help="maximum channel pruning ratio")
    parser.add_argument("--sl-epochs", type=int, default=None)
    parser.add_argument("--sl-resume", type=str, default=None)
    parser.add_argument("--sl-lr", default=None, type=float, help="learning rate")
    parser.add_argument("--sl-lr-step-size", default=None, type=int, help="milestones for learning rate decay")
    parser.add_argument("--sl-lr-warmup-epochs", default=None, type=int, help="the number of epochs to warmup (default: 0)")
    
    
    # my settings
    parser.add_argument("--N_batchs", default=50, type=int, help="how many batches to accumulate Jacobian")
    parser.add_argument("--Jacobian_batch_size", default=None, type=int, help="if not None, use this bs only in evaluating Jacobian; just a small bs limited by CUDA memory")
    parser.add_argument("--group_reduction", default='sum', type=str)
    parser.add_argument("--normalizer", default=None, type=str)
    parser.add_argument("--equal_pruning", default=False, action="store_true")
    parser.add_argument("--train_naive_pruning", default=False, action="store_true")
    
    parser.add_argument("--wd_div", default=1, type=float, help='divide factor for weight_decay of Compressor and Decompressor (C&D); if 0, weight_decay=0 for C&D')
    parser.add_argument("--lr_div", default=1, type=float, help='divide factor for lr of Compressor and Decompressor (C&D); if 0, lr=0 for C&D')
    
    parser.add_argument("--sparsity_learning", default=False, action="store_true")
    parser.add_argument("--pruning_idx_resume_path", default=None, type=str)
    parser.add_argument("--resnet_skip_block_last", default=False, action="store_true", help='dont no prune the last layer of each block')
    parser.add_argument("--mobile_ignore_last", default=False, action="store_true", help='ignore features.18.1 or not')
    parser.add_argument("--Dropout", default=0.2, type=float, help='Set the drop out for classifier')

    ## distillation
    parser.add_argument("--distill", default=False, action="store_true")
    parser.add_argument("--coeff-ce", default=0.5, type=float) # cross entropy loss
    parser.add_argument("--coeff-label", default=0.5, type=float) # label distillation loss
    parser.add_argument("--T", default=4, type=float) # label distillation temperature
    
    parser.add_argument("--isomorphic", default=False, action="store_true", help='isomorphic or not')



    return parser





class Compress(torch.nn.Module):
    
    def __init__(self, orig_layer):
        super(Compress, self).__init__()

        if isinstance(orig_layer, torch.nn.Linear):
            dim = orig_layer.out_features
            self.Compressor = nn.Linear(dim, dim, bias=False)
            self.orig_layer = orig_layer
            self.position = 'after'            
        elif isinstance(orig_layer, torch.nn.BatchNorm2d):
            dim = orig_layer.num_features
            self.position = 'before'
            self.Compressor = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
            self.orig_layer = orig_layer  
        elif isinstance(orig_layer, torch.nn.Conv2d):
            self.position = 'after'
            dim = orig_layer.out_channel
            self.orig_layer = orig_layer  
            self.Compressor = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        else:
            raise NotImplementedError
           
        nn.init.eye_(self.Compressor.weight.data.squeeze())

    def forward(self, x):
        if self.position=='before': # Compressor-OriginalBN
            x = self.Compressor(x)
            x = self.orig_layer(x)
        elif self.position=='after': # OriginalConv-Compressor
            x = self.orig_layer(x)
            x = self.Compressor(x)
        return x 

class Decompress(torch.nn.Module):
    def __init__(self, orig_layer):
        super(Decompress, self).__init__()
        
        if isinstance(orig_layer, torch.nn.Linear):
           self.Decompressor = nn.Linear(orig_layer.in_features, orig_layer.in_features, bias=False) 
        elif isinstance(orig_layer, torch.nn.Conv2d):
            self.Decompressor = nn.Conv2d(orig_layer.in_channels, orig_layer.in_channels, 1, 1, 0, bias=False)
        else:
            raise NotImplementedError
        
        nn.init.eye_(self.Decompressor.weight.data.squeeze())
        self.orig_layer = orig_layer
        
    def forward(self, x):
        x = self.Decompressor(x)
        x = self.orig_layer(x)
        return x 



# Function to modify ResNet
@torch.no_grad()
def C_D_resnet(model):
    """
    Replaces all Conv2d and BatchNorm2d layers in the ResNet model
    according to the specified rules.
    """
    for name, module in model.named_modules():
        
        if args.resnet_skip_block_last and (name=='conv1' or 'conv1' in name or 'down' in name or 'fc' in name):
            continue
   
        # Decompressor (except the first one, as in_channels=3)
        if isinstance(module, nn.Conv2d) and module.in_channels != 3:
            
            parent_name = ".".join(name.split(".")[:-1])  # Get the parent module's name
            module_name = name.split(".")[-1]  # Get the current module's name
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]
            # args.logger.info('Decompressor parent_name',parent_name, '; module_name', module_name )
            # Replace the Conv2d layer
            setattr(parent_module, module_name, Decompress(module))
            
        # Decompressor for Linear layer
        elif isinstance(module, torch.nn.Linear):
            assert (name=='fc')
            setattr(model, name,  Decompress(module))  

    # Compressor
    for name, module in model.named_modules():
        
        if args.resnet_skip_block_last and (name=='bn1' or 'bn3' in name or 'down' in name):
            continue
        
        if isinstance(module, nn.BatchNorm2d):
            
            parent_name = ".".join(name.split(".")[:-1])  # Get the parent module's name
            module_name = name.split(".")[-1]  # Get the current module's name
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]

            # Replace the BatchNorm2d layer
            setattr(parent_module, module_name, Compress(module))




# Function to modify vit
@torch.no_grad()
def C_D_vit(model):
    """
    Replaces all mlp.0 layer in vit
    according to the specified rules.
    """
    
    for name, module in model.named_modules():
        
        # Compressor
        if 'mlp.0' in name:
            
            parent_name = ".".join(name.split(".")[:-1])  # Get the parent module's name
            module_name = name.split(".")[-1]  # Get the current module's name
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]

            # Replace the BatchNorm2d layer
            setattr(parent_module, module_name, Compress(module))
        
        # Decompressor
        if 'mlp.3' in name:
            
            parent_name = ".".join(name.split(".")[:-1])  # Get the parent module's name
            module_name = name.split(".")[-1]  # Get the current module's name
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]

            # Replace the BatchNorm2d layer
            setattr(parent_module, module_name, Decompress(module))      





# 自定义函数：根据层类型和属性筛选参数
def get_param_groups(model, pruner, lr, weight_decay):
    one_by_one_conv_params = []
    other_params = []
    
    for m_name, module in model.named_modules():
        
        if m_name.endswith('.Compressor') or m_name.endswith('.Decompressor'): 
            # compressor or decompressor for pruning each conv1 of basicblock   
            for name, param in module.named_parameters(recurse=False):
                one_by_one_conv_params.append(param)
                print0(f'one_by_one_conv_params: {m_name}.{name}')

        else:
            for name, param in module.named_parameters(recurse=False):
                other_params.append(param)
                
    if one_by_one_conv_params==[]:
        print0(f'\n one_by_one_conv_params==[]\n')
        return [{"params": other_params, "lr": lr, "weight_decay": weight_decay if pruner is None else 0}]
    else:
        return [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay if pruner is None else 0},
            {"params": one_by_one_conv_params, "lr": lr/args.lr_div if args.lr_div!=0 else 0, "weight_decay": weight_decay/args.wd_div if args.wd_div!=0 else 0},
        ]


# modify from utils.set_weight_decay
# set differeent divide factor for weight_decay and lr for Compressor and Decompressor
def set_weight_decay_EP(
    model: torch.nn.Module,
    weight_decay: float,
    lr: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
    args=None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
        "EP": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
        "EP": 0 if args and args.wd_div==0 else weight_decay/args.wd_div,
    }
    
    lr_EP = 0 if args and args.lr_div==0 else lr/args.lr_div 
    
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix="", D_C=False):
        if D_C:
            for n, m in module.named_parameters():
            # 'Decompressor' in name or 'Compressor' in name:
                params["EP"].append(m)
                print0(f'EP_ parameters: {n}')
                # input(f'EP_ parameters: {name}')
                # continue
            return
                
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
        
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            # print0(child_name)
            _add_params(child_module, prefix=child_prefix, D_C='Decompressor' in child_name or 'Compressor' in child_name)
        # print0('finish printing')

    _add_params(model)

    param_groups = []
    for key in params:
        if 'EP' in key:
            if len(params[key]) > 0:
                param_groups.append({"params": params[key], "lr": lr_EP, "weight_decay": params_weight_decay[key]})
        else:
            if len(params[key]) > 0:
                param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups





def get_pruning_idxs(model, example_inputs, train_loader, args, target_flops, data_loader_test):
    
    """
    Only do Jacobian pruning on the main process. Save the pruned model.
    """

    proxy_model = copy.deepcopy(model)
    proxy_model.to(args.device)
    example_inputs = example_inputs.to(args.device)
    proxy_model.eval()
    
    # 执行Jacobian剪枝
    pruner = get_pruner(proxy_model, example_inputs, args, default_depgraph=False)
    
    if not 'Jacobian' in args.method: # Only consider my Jacobian method
        raise NotImplementedError
    
    else:
        ori_ops, orig_parameters = tp.utils.count_ops_and_params(proxy_model, example_inputs=example_inputs)
        print0(f'{"="*20}original model: ops {ori_ops/1e9} G; paras {orig_parameters/1e6} M')
        
        remain_ops = ori_ops
        pruning_record_orig_wo = {}
        step_count = 0
        
        while remain_ops / 1e9 > target_flops:# keep pruning
            torch.cuda.empty_cache()
            
            start_time = time.time()
            # data-driven Jacobian Criteria
            proxy_model.eval()
            proxy_model.zero_grad()
            imp = pruner.importance
            imp.zero_grad()
            imp.zero_score()
            N_batchs = args.N_batchs
            assert(N_batchs>0)
            for k, (imgs, lbls) in enumerate(train_loader):
                if k>=N_batchs: break
                imgs = imgs.to(args.device)
                lbls = lbls.to(args.device)
                output = proxy_model(imgs)
                loss = torch.nn.functional.cross_entropy(output, lbls)
                proxy_model.zero_grad() # clear gradients
                loss.backward()
                imp.accumulate_grad(proxy_model) # accumulate Jacobian                       
                if (k+1)%10==0 or k+1==N_batchs: # accumulate score every x batches (and clean grad) in case CUDA OUT OF MEMORY
                    imp.accumulate_score(proxy_model)
                    torch.cuda.empty_cache()

            for id, group in enumerate(pruner.step(interactive=True)): 
                dep, deleted_idxs = group[0]
                target_module = dep.target.module
                layer_name = dep.target.name[:dep.target.name.find(' ')] # Get the layer name
                if layer_name not in pruning_record_orig_wo:
                    if hasattr(target_module, 'out_channels'):
                        pruning_record_orig_wo[layer_name] = [np.arange(target_module.out_channels), np.arange(target_module.out_channels)]
                    elif hasattr(target_module, 'out_features'):
                        pruning_record_orig_wo[layer_name] = [np.arange(target_module.out_features), np.arange(target_module.out_features)]
                    else:
                        raise NotImplementedError
                        
                # runing_record_orig_wo[layer_name][0] is the original full index
                # runing_record_orig_wo[layer_name][1] is the remained index; 
                pruning_record_orig_wo[layer_name][1] = np.delete(pruning_record_orig_wo[layer_name][1], deleted_idxs)
                # prune here
                group.prune()   
                    
            if 'vit' in args.model:
                proxy_model.hidden_dim = proxy_model.conv_proj.out_channels
            
            end_time = time.time()
            remain_ops, remain_parameters = tp.utils.count_ops_and_params(proxy_model, example_inputs=example_inputs)
            print0(f"step {step_count}: ops {remain_ops/ 1e9} G; target {target_flops} G; Speedup {float(ori_ops) /(remain_ops/ 1e9)}x; paras {remain_parameters / 1e6} M, rate {remain_parameters / orig_parameters * 100}%; time {(end_time-start_time)/60:.2f} mins")
            step_count += 1
        
        
        # pruned_idxs = all_idxs - remained_idxs
        print0(f'\n{"="*20}Pruning results...')
        for item in pruning_record_orig_wo:
            print0(f'{item}, #Original filter={len(pruning_record_orig_wo[item][0])}, #Remain={len(pruning_record_orig_wo[item][1])}, #Pruned={len(pruning_record_orig_wo[item][0])-len(pruning_record_orig_wo[item][1])}')
            pruning_record_orig_wo[item] = np.setdiff1d(pruning_record_orig_wo[item][0], pruning_record_orig_wo[item][1])

        print0(f"{'='*30}raw model after pruning")
        print0(proxy_model) if not args.train_naive_pruning else None
        print0("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(orig_parameters / 1e6, remain_parameters / 1e6, remain_parameters / orig_parameters * 100))
        print0("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(ori_ops / 1e9, remain_ops / 1e9, remain_ops / ori_ops * 100, ori_ops / ori_ops))
        print0("="*16)
        
        # evaluate(proxy_model, nn.CrossEntropyLoss(), data_loader_test, device=args.device)

    if dist.get_rank() == 0:
        torch.save(proxy_model, args.output_dir+'/raw_Naive_pruned_model.pth')
        
    del pruner    
    del proxy_model
    torch.cuda.empty_cache()
    
    return pruning_record_orig_wo

def prune_model(model, pruning_record_orig_wo, example_inputs, EP=False):
    
    if not EP: # prune the original model
        model.eval()
        DG = tp.DependencyGraph().build_dependency(model, example_inputs)
        for name, module in model.named_modules():

            if name in pruning_record_orig_wo:
                idxs = pruning_record_orig_wo[name]
                if hasattr(module, 'out_channels'):
                    group = DG.get_pruning_group(module,  tp.prune_conv_out_channels, idxs)  
                elif hasattr(module, 'out_features'):
                    group = DG.get_pruning_group(module,  tp.prune_linear_out_channels, idxs) 
                group.prune()        
        
    else: # prune the equivalent pruning model
        model.eval()
        DG = tp.DependencyGraph().build_dependency(model, example_inputs)
        print(model)
        if 'resnet' in args.model:
            for name, module in model.named_modules():
                if 'Compressor' in name:
                    assert isinstance(module, nn.Conv2d)
                    name = name.replace('.Compressor','')
                    if 'down' in name:
                        assert(name.endswith('.1'))
                        name = name.replace('.1', '.0') 
                    else:
                        name = name.replace('bn', 'conv')
                            
                    if name in pruning_record_orig_wo:
                        idxs = pruning_record_orig_wo[name]
                        group = DG.get_pruning_group(module,  tp.prune_conv_out_channels, idxs)
                        group.prune()  
            
            # delete those unpruned Compressor-Decompressor
            for name, module in model.named_modules():
                if 'ompressor' in name and ((isinstance(module, torch.nn.Conv2d) and module.in_channels==module.out_channels) \
                    or (isinstance(module, torch.nn.Linear) and module.in_features==module.out_features)):
                        
                        parent_module = model
                        module_names = name.split(".")  
                        
                        for sub_name in module_names[:-1]:
                            parent_module = getattr(parent_module, sub_name)  
                        
                        # Identity
                        setattr(parent_module, module_names[-1], nn.Identity())
                        print0(f"Replaced {name} with nn.Identity")
        elif 'vit' in args.model:
            for name, idxs in pruning_record_orig_wo.items():
                assert 'mlp' in name, 'Hey, this pruned layer is not mlp, check it now'
                if name.endswith('.0'):
                    name = name+'.Compressor'
                elif name.endswith('.3'):
                    name = name+'.Decompressor'
                else:
                    print(name)
                    raise NotImplementedError
                
                module = dict(model.named_modules())[name]
                group = DG.get_pruning_group(module,  tp.prune_linear_out_channels, idxs)
                group.prune()  
        
        
        
    if 'vit' in args.model:
        model.hidden_dim = model.conv_proj.out_channels                 
                    

    
def prune_to_target_flops(pruner, model, target_flops, example_inputs):

    model.eval()
    ori_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    pruned_ops = ori_ops
    while pruned_ops / 1e9 > target_flops:
        pruner.step()
        if 'vit' in args.model:
            model.hidden_dim = model.conv_proj.out_channels
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        
    return pruned_ops

def get_pruner(model, example_inputs, args, default_depgraph=False):
    
    # This one (Default of Depgraph is not working probably due to the update of Troch-pruning)
    # unwrapped_parameters = (
    #     [model.encoder.pos_embedding, model.class_token] if "vit" in args.model else None
    # )
    
    # So I modify it as 
    unwrapped_parameters = [
        (model.encoder.pos_embedding,0), (model.class_token,0) 
    ] if "vit" in args.model else None
    # unwrapped_parameters = None
    
    sparsity_learning = False
    data_dependency = False
    
    
    if default_depgraph:
        # sparsity_learning = True
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)   
    else:
        
        if args.method == "random":
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "l1":
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "lamp":
            imp = tp.importance.LAMPImportance(p=2)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "slim":
            sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
        elif args.method == "group_norm":
            imp = tp.importance.GroupMagnitudeImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
        elif args.method == "group_greg":
            sparsity_learning = True
            imp = tp.importance.GroupMagnitudeImportance(p=2)
            pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
        elif args.method == "group_sl":
            sparsity_learning = True
            imp = tp.importance.GroupMagnitudeImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
        elif args.method == "Jacobian" or 'jaco' in args.method:
            imp = GroupJacobianImportance_accumulate(group_reduction=args.group_reduction, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MetaPruner, global_pruning=args.global_pruning)
        else:
            raise NotImplementedError
        
    args.data_dependency = data_dependency
    # args.c = sparsity_learning
    ignored_layers = []
    pruning_ratio_dict = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
        
        if args.resnet_skip_block_last:
            if isinstance(m, torch.nn.Conv2d) and ('conv3' in n or n=='conv1' or 'down' in n):
                ignored_layers.append(m)
                
    if 'mobile' in args.model and args.mobile_ignore_last:
        ignored_layers.append(model.features[18][0])  
    
    round_to = None
    if 'vit' in args.model:
        round_to = model.encoder.layers[0].num_heads
        
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=100,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=args.max_pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        isomorphic=args.isomorphic
    )
    return pruner



def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, pruner=None, recover=None,  teacher_model=None):
    if teacher_model is not None:
        teacher_model.eval()
        
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        start_time = time.time()
        image, target = image.to(device), target.to(device)
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast(device_type=args.device, enabled=scaler is not None):
            if teacher_model is not None:
                output = model(image)
                with torch.no_grad():
                    output_teacher = teacher_model(image)
                         
                kd_loss =  torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(output/args.T, dim=1), # Student predictions (softened)
                    torch.nn.functional.softmax(output_teacher/args.T, dim=1), # Teacher predictions (softened)
                    reduction='batchmean' # Note that default is 'mean', which is divided by the number of elements in the output
                    ) *   (args.T ** 2) # Scale by temperature squared
                
                loss_ce = criterion(output, target)
                loss = args.coeff_label * kd_loss + args.coeff_ce * loss_ce
                
            else:
        
                output = model(image)
                loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if pruner:
                scaler.unscale_(optimizer)
                pruner.regularize(model)
            #if recover:
            #    recover(model.module)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if pruner is not None:
                pruner.regularize(model)
            if recover:
                recover(model.module)
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        
    if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
        pruner.update_reg()

def  evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    
    num_processed_samples = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print0(f"{'='*20}{header} Acc@1 {metric_logger.acc1.global_avg:.3f};  Acc@5 {metric_logger.acc5.global_avg:.3f};  loss {metric_logger.loss.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print0("Loading data...")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)

    print0("Loading training data...")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print0("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path,weights_only=False)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                              random_erase_prob=random_erase_prob))
        if args.cache_dataset:
            print0("Saving dataset_train to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print0("Data loading took", time.time() - st)

    print0("Loading validation data...")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print0("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path,weights_only=False)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))
        if args.cache_dataset:
            print0("Saving dataset_test to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print0("Creating data loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def flatten_dict(dic):
    flattned = dict()
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'/%s'%k, v )
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v
        
    _flatten(None, dic)
    return flattned

def get_logger(name='train', output=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # STDOUT
    stdout_handler = logging.StreamHandler( stream=sys.stdout )
    stdout_handler.setLevel( logging.DEBUG )

    plain_formatter = logging.Formatter( 
            "[%(asctime)s]: %(message)s", datefmt="%m/%d %H:%M:%S" )
    formatter = plain_formatter
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    # FILE
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            filename = output
        else:
            os.makedirs(output, exist_ok=True)
            filename = os.path.join(output, "log.txt")
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger
    
def printf_on_rank0(*infos, logger=None):
    rank = dist.get_rank() if dist.is_initialized() else 0  # 默认rank 0
    
    if rank == 0:
        # if isinstance(info, argparse.Namespace):
        #     info = vars(info)
        # if not isinstance(info, str):
        #     info = json.dumps(info, indent=2)
        # message = " ".join(map(str, info))
        for info in infos:
            logger.info(info)
    # dist.barrier()
    
def main(args):
    # if args.output_dir:
        
    utils.init_distributed_mode(args)
    
    middle_name = ''
    middle_name += 'isomorphic' if args.isomorphic else '' 
    middle_name += '_SkipLast' if args.resnet_skip_block_last or args.mobile_ignore_last else ''
    middle_name += f'_Dropout{args.Dropout}'
    middle_name += '_Sparse' if args.sparsity_learning else ''
    middle_name += '_EP' if args.equal_pruning else ''
    if args.distill:
        args.output_dir = f'{args.output_dir}_Distill_{args.coeff_ce}_{args.coeff_label}_{args.T}_' 
    # if args.mobile_ignore_last:
    #      args.output_dir += 'no_last'
    if args.normalizer=='None': args.normalizer = None 
    
    args.output_dir = f'results/imagenet/{args.model}/{args.output_dir}{args.method}_Gr{args.group_reduction}_No{args.normalizer}_{middle_name}_{args.target_flops}G_Nbatch{args.N_batchs}_lr{args.lr}_lrD{args.lr_div}_wdD{args.wd_div}_epoch{args.epochs}'
    if args.amp:
        args.output_dir += '_amp'
    
    if dist.get_rank() == 0:
        utils.mkdir(args.output_dir)
        logger = get_logger(output=args.output_dir+'/log.log')
    else:
        logger = None
    
    global print0
    print0 = partial(printf_on_rank0, logger=logger)
    
    for k, v in flatten_dict(vars(args)).items():  # args.logger.info args
        print0("%s: %s" % (k, v))
    

    device = torch.device(args.device)
    # args.device = device

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, # if args.Jacobian_batch_size==None else args.Jacobian_batch_size
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=250, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    print0("Creating model")
    model = registry.get_model(num_classes=1000, name=args.model, pretrained=args.pretrained, target_dataset='imagenet') #torchvision.models.__dict__[args.model](pretrained=args.pretrained) #torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    if 'mobil' in args.model:
        model.classifier[0] = nn.Dropout(args.Dropout)
    model.eval()
    print0("="*16)
    print0(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    print0("Params: {:.4f} M".format(base_params / 1e6))
    print0("ops: {:.4f} G".format(base_ops / 1e9))
    print0("="*16)
    if args.prune:
        
                
        
        if args.sparsity_learning:
            if args.sl_resume:
                print0("Loading sparse model from {}...".format(args.sl_resume))
                model.load_state_dict( torch.load(args.sl_resume, map_location='cpu',weights_only=False)['model'] )
            else:
                pruner = get_pruner(model, example_inputs=example_inputs, args=args, default_depgraph=True)
                print0("Sparsifying model...")
                if args.sl_lr is None: args.sl_lr = args.lr
                if args.sl_lr_step_size is None: args.sl_lr_step_size = args.lr_step_size
                if args.sl_lr_warmup_epochs is None: args.sl_lr_warmup_epochs = args.lr_warmup_epochs
                if args.sl_epochs is None: args.sl_epochs = args.epochs
                model = train(model, args.sl_epochs, 
                                        lr=args.sl_lr, lr_step_size=args.sl_lr_step_size, lr_warmup_epochs=args.sl_lr_warmup_epochs, 
                                        train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                                        device=device, args=args, pruner=pruner, state_dict_only=True, words='Sparse_')
                #model.load_state_dict( torch.load('regularized_{:.4f}_best.pth'.format(args.reg), map_location='cpu')['model'] )
                #utils.save_on_master(
                #    model_without_ddp.state_dict(),
                #    os.path.join(args.output_dir, 'regularized-{:.4f}.pth'.format(args.reg)))
   
                del pruner
                torch.cuda.empty_cache() 
                dist.barrier()
        
        ###############################################
        # Jacobian pruning in the main process, getting the pruning record
        # and then broadcast the pruned record to all processes and prune model
        ###############################################        
        # proxy_model = None # initialize as {} for all rank


        utils.save_on_master(model, args.output_dir+"/tmp_model_for_prune.pth")
        dist.barrier()
        
        if args.pruning_idx_resume_path==None:
            idxs_path = args.output_dir + "/pruning_record_orig_wo.pth"
            print0(f"\n{'*'*30}Getting Pruning Idxs in rank 0...")
        else:
            idxs_path = args.pruning_idx_resume_path
            print0(f"\n{'*'*30}Loading Pruning Idxs from {idxs_path}...")
        
        

                    
        if utils.is_main_process() and args.pruning_idx_resume_path==None:
            data_loader_pruning = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size if args.Jacobian_batch_size==None else args.Jacobian_batch_size,
                sampler=train_sampler,
                num_workers=args.workers,
                pin_memory=True,
                collate_fn=collate_fn)
                    
            start_time = time.time()
            pruning_record_orig_wo = get_pruning_idxs(model, example_inputs, data_loader_pruning, args, args.target_flops, data_loader_test)
            end_time = time.time()
            print0(f'Time cost: {(end_time-start_time)/60} mins')
            
            torch.save(pruning_record_orig_wo, idxs_path)
            if os.path.exists(idxs_path):
                print0(f"File {idxs_path} saved successfully.")
            else:
                print0(f"File {idxs_path} saving failed.")
                
            del data_loader_pruning
        
        # the `timeout` is set to be 3hour in utils
        dist.barrier() # Synchronize all ranks; all processes should be in barrier then they will go next; so other ranks will wait rank0 to go here
        
        
        
        # load for all GPU/Process/Rank
        pruning_record_orig_wo = torch.load(idxs_path, map_location='cpu',weights_only=False)
        assert pruning_record_orig_wo!={}, 'pruning_record_orig_wo is empty'
        
        dist.barrier()
        
        torch.cuda.empty_cache() # save cuda memory, due to my poor hardward  /(ㄒoㄒ)/~~
        
        ###############################################
        # Equivalent pruning
        ###############################################  
        if args.equal_pruning: # EP
            print0(f'\n {"="*30} EP Creating {"="*30} \n')
            if 'resnet' in  args.model:
                C_D_resnet(model) 
            elif 'vit' in args.model:
                C_D_vit(model)
            else:
                raise NotImplementedError
            
            # print0(f'\n {"="*30} EP Pruning {"="*30} \n')
            print0("Pruning model...")
            model = model.to('cpu')
            prune_model(model, pruning_record_orig_wo, example_inputs, EP=args.equal_pruning)
 
            model.eval()
            pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            # print0('\n', "="*16)
            print0("After pruning:")
            print0(model)
            print0("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
            print0("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
            print0(f'gpu {args.gpu}; device {device}; get_rank {dist.get_rank()}')   
                
            words = 'EP_'
            print0('\n', "="*16)
            dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
            print0("Finetuning..." if args.prune else "Training...")
            torch.cuda.empty_cache()
            train(model, args.epochs, 
                    lr=args.lr, lr_step_size=args.lr_step_size, lr_warmup_epochs=args.lr_warmup_epochs, 
                    train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                    device=device, args=args, pruner=None, state_dict_only=(not args.prune), words=words)
            
            del model 
            torch.cuda.empty_cache() # save cuda memory
            
            
        if args.train_naive_pruning: # naive pruning
            print0(f'\n {"="*30} Naive Pruning {"="*30} \n')
            
            
            # model_path = args.output_dir+'/raw_Naive_pruned_model.pth'
            # model_path_tmp = args.output_dir + '/tmp_model_for_prune.pth'
            # if os.path.exists(model_path):
            #     model = torch.load(model_path, map_location='cpu')
            # elif os.path.exists(model_path_tmp):
            #     model = torch.load(model_path_tmp, map_location='cpu')
            # else:
            #     print0('NO PATHS for Pruned/Unpruned Model for ')
                            
            model_path_tmp = args.output_dir + '/tmp_model_for_prune.pth'
            if os.path.exists(model_path_tmp):
                model = torch.load(model_path_tmp, map_location='cpu',weights_only=False)
            else:
                print0('PATHS for Pruned/Unpruned Model not exist!')
            
            model = model.to('cpu')
            prune_model(model, pruning_record_orig_wo, example_inputs)
            
            model = model.to('cpu')
            model.eval()
            pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            print0('\n', "="*16)
            print0("After pruning:")
            print0(model)
            print0("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
            print0("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
                      
            words = 'Naive_'
            print0('\n', "="*16)
            dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
            print0("Finetuning..." if args.prune else "Training...")
            train(model, args.epochs, 
                    lr=args.lr, lr_step_size=args.lr_step_size, lr_warmup_epochs=args.lr_warmup_epochs, 
                    train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                    device=device, args=args, pruner=None, state_dict_only=(not args.prune), words=words)
    

def train(
    model, 
    epochs, 
    lr, lr_step_size, lr_warmup_epochs, 
    train_sampler, data_loader, data_loader_test, 
    device, args, pruner=None, state_dict_only=True, recover=None, words=''):
    
    writer_words = 'naive' if words=='' else words.replace('_', '')
    writer = SummaryWriter(args.output_dir+f'/{writer_words}')
    
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.label_smoothing>0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    weight_decay = args.weight_decay if pruner is None else 0
    bias_weight_decay = args.bias_weight_decay if pruner is None else 0
    norm_weight_decay = args.norm_weight_decay if pruner is None else 0

    custom_keys_weight_decay = []
    if bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
            
    # parameters = utils.set_weight_decay(
    parameters = set_weight_decay_EP(
        model,
        weight_decay,
        lr,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        args=args,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=args.momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    # scaler = torch.cuda.amp.GradScaler() if args.amp else None
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
        
    if args.distill:
        teacher_model = registry.get_model(num_classes=1000, name=args.model, pretrained=args.pretrained, target_dataset='imagenet') #torchvision.models.__dict__[args.model](pretrained=args.pretrained) #torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
        teacher_model.eval()
        teacher_model.to(device)
        if args.distributed:
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu])
            teacher_model.eval()
    else:
        teacher_model = None



    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return
    
    start_time = time.time()
    best_acc = 0
    prefix = '' if pruner is None else 'regularized_{:e}_'.format(args.reg)

    print0(f'{"="*10}{words} raw accuracy before training{"="*10}')
    acc = evaluate(model, criterion, data_loader_test, device=device)
    print0(f'\nTrainning starts...') 
    

        
    for epoch in range(args.start_epoch, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, pruner, recover=recover, teacher_model=teacher_model)
        lr_scheduler.step()
        acc = evaluate(model, criterion, data_loader_test, device=device)
        writer.add_scalar('acc', acc, epoch)
        
        if model_ema:
            acc = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            writer.add_scalar('acc_ema', acc, epoch)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict() if state_dict_only else model_without_ddp,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if acc>best_acc:
                best_acc=acc
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, words+prefix+"best.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, words+prefix+"latest.pth"))
        print0("Epoch {}/{}, Current Best Acc = {:.6f}\n".format(epoch, epochs, best_acc))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print0(f"Training time {total_time_str}")
    return model_without_ddp

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    dist.destroy_process_group()



