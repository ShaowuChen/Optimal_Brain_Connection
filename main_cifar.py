''' 
For CIFAR: VGG/ResNet
'''

import sys, os

from functools import partial
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
import engine.utils as utils
import registry

# my implement
from Selfmake_Importance import GroupJacobianImportance_accumulate, WHCImportance
import copy
import numpy as np
from utils_this_project import save_dir_name_pruned, check_and_add_row
import re

import logging

from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "prune", "test"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="cifar100", choices=['cifar10', 'cifar100', 'modelnet40'])
parser.add_argument('--dataroot', default='/data/csw/dataset', help='path to your datasets')
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--total-epochs", type=int, default=100)
parser.add_argument("--lr-decay-milestones", default="60,80", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--restore", type=str, default='./results/base_model/DepGraph')
parser.add_argument('--output-dir', default='TestV2', help='path where to save')
parser.add_argument("--finetune", action="store_true", default=False, help='whether finetune or not')

# For pruning
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--speed-up", type=float, default=2)
parser.add_argument("--max-pruning-ratio", type=float, default=1)
parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
parser.add_argument("--reg", type=float, default=5e-4)
parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
parser.add_argument("--weight-decay", type=float, default=5e-4)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=False)
parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
parser.add_argument("--sl-restore", type=str, default=None)
parser.add_argument("--iterative-steps", default=400, type=int)

# my implementation
parser.add_argument("--sparsity-learning", default=False, action="store_true") # usiong Equivalent pruning or not 

parser.add_argument("--equivalent", default=False, action="store_true") # usiong Equivalent pruning or not 
parser.add_argument("--group-reduction", default='mean', type=str)  
parser.add_argument("--normalizer", default=None, type=str)  
parser.add_argument("--group_reduction",default='mean', type=str)

parser.add_argument("--N-batchs", default=-1, type=int, help='if -1, use all batches') 
parser.add_argument("--delete-rate", default=0, type=float) # pruned with a given rate 
parser.add_argument("--save-suffix", default=0, choices=[0,1,2], type=int) # pruned with a given rate 


# for finetuning; not used currently
parser.add_argument("--coeff-ce", default=1, type=float) # cross entropy loss
parser.add_argument("--coeff-label", default=0, type=float) # label distillation loss
parser.add_argument("--T", default=4, type=float) # label distillation temperature

# my implementation; for finetuning; not used currently
parser.add_argument("--expand-rate", default=0, type=float) # pruned with a given rate 
parser.add_argument("--initi-div", default=10, type=float) # pruned with a given rate

parser.add_argument("--warmup", default="False", type=str)

parser.add_argument("--print", default=False, action="store_true")

parser.add_argument("--wd-div", default=1, type=float)
parser.add_argument("--lr-div", default=1, type=float)


parser.add_argument("--resnet_conv1_only", default=False, action="store_true")
parser.add_argument("--randomC", default=False, action="store_true")
parser.add_argument("--randomD", default=False, action="store_true")
parser.add_argument("--freeze_old_bn", default=False, action="store_true")

parser.add_argument("--freeze_c2", default=False, action="store_true", help='freeze compressor, new bn and decompressor of each second conv for a while')
parser.add_argument("--freeze_c1", default=False, action="store_true", help='freeze compressor, new bn and decompressor of each second conv for a whil')


parser.add_argument("--select", required=True, choices=['BC', 'CB', 'BCB', 'BCB_replace', 'BCB_momentumNone'])

parser.add_argument("--normalizer_for_sl",default='max', type=str)

parser.add_argument("--first_layer_lr_div",default=None, type=float)


args = parser.parse_args()




@torch.no_grad()
def Compressor_bcb(bn_module):
    
    C_conv = torch.nn.Conv2d(bn_module.num_features, bn_module.num_features, 1, 1, 0, bias=False)
    # initialize diagnonal elements as 1
    nn.init.eye_(C_conv.weight.data.squeeze())

    if args.randomC:
        # creeat random_tensor and initialized using kaiming_normal_
        random_tensor = torch.zeros_like(C_conv.weight.data)
        nn.init.kaiming_normal_(random_tensor)
        random_tensor /= args.initi_div

        # replace non-diagonal elements
        with torch.no_grad():
            weight_data = C_conv.weight.data.squeeze()  
            mask = torch.eye(weight_data.size(0), device=weight_data.device)  # diagonal mask
            weight_data *= mask  # keep diagonal elements
            assert(torch.any(weight_data==C_conv.weight.data.squeeze()))
            weight_data += (1 - mask) * random_tensor.squeeze()  # replace non-diagonal elements 
        C_conv.weight.data.copy_(weight_data.unsqueeze(2).unsqueeze(3))  # restore original shape  


    if args.select=='BC':
        return torch.nn.Sequential(bn_module, C_conv, torch.nn.Identity())
    
    elif args.select=='CB':
        return torch.nn.Sequential(torch.nn.Identity(), C_conv,  bn_module)
    
    elif args.select=='BCB' or args.select=='BCB_replace':
        C_bn = torch.nn.BatchNorm2d(bn_module.num_features)
           
        if args.select=='BCB_replace':
            # pass
            C_bn.weight.copy_(bn_module.weight.data)
            C_bn.bias.copy_(bn_module.bias.data)
            C_bn.running_var.copy_(bn_module.running_var.data)
            C_bn.running_mean.copy_(bn_module.running_mean.data) 
            C_bn.eps = bn_module.eps
            
            # bn_module.weight.data = torch.ones_like(bn_module.weight.data)
            # bn_module.bias.data = torch.zeros_like(bn_module.bias.data)
            # bn_module.running_var.data = torch.ones_like(bn_module.running_var.data)
            # bn_module.running_mean.data = torch.zeros_like(bn_module.running_mean.data)
            # bn_module.eps = 0
        return torch.nn.Sequential(bn_module, C_conv, C_bn)
    
    elif args.select=='BCB_momentumNone':
        C_bn = torch.nn.BatchNorm2d(bn_module.num_features, momentum=None) # then it would use 1/num_batches_tracked as momentum
        return torch.nn.Sequential(bn_module, C_conv, C_bn)
        
    else:
        raise NotImplementedError

    

# replace the original_conv with a {1x1_conv, original_conv} module
@torch.no_grad()
def Decompressor(conv_module):
    
    D_conv = torch.nn.Conv2d(conv_module.in_channels, conv_module.in_channels, 1, 1, 0, bias=False)
    nn.init.eye_(D_conv.weight.data.squeeze())

    if args.randomD:
        # create random_tensor and kaiming_normal_ 
        random_tensor = torch.zeros_like(D_conv.weight.data)
        nn.init.kaiming_normal_(random_tensor)
        random_tensor /= args.initi_div

        with torch.no_grad():
            weight_data = D_conv.weight.data.squeeze()  
            mask = torch.eye(weight_data.size(0), device=weight_data.device)  # diagonal mask
            weight_data *= mask  # keep diagonal elements
            assert(torch.any(weight_data==D_conv.weight.data.squeeze()))
            weight_data += (1 - mask) * random_tensor.squeeze()  # replace non-diagonal elements 
            # reassign to D_conv.weight
        D_conv.weight.data.copy_(weight_data.unsqueeze(2).unsqueeze(3))  # restore original shape
    
    return torch.nn.Sequential(D_conv, conv_module)


@torch.no_grad()
def C_D_vgg(model):
    for name, module in model.named_modules():
        # Decompressor for Conv layer
        if isinstance(module, torch.nn.Conv2d) and module.in_channels!=3:
            parent_name = ".".join(name.split(".")[:-1])  
            module_name = name.split(".")[-1]  
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]
            
            setattr(parent_module, module_name, Decompressor(module))  
        
        # Decompressor for Linear layer
        elif isinstance(module, torch.nn.Linear):
            assert (name=='classifier')
            setattr(model, name, nn.Sequential(nn.Linear(module.in_features,module.in_features, bias=False), module))  
            nn.init.eye_(model.classifier[0].weight.data) # identity initialization
    
    # Compressor_bcb
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            parent_name = ".".join(name.split(".")[:-1])  
            module_name = name.split(".")[-1] 
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]
            
            setattr(parent_module, module_name, Compressor_bcb(module))



def check_and_modify_vgg(input_str):
    # check if blocka.b.c format, return blocka.b-1 if true, else return None

    pattern = r'^block(\d+)\.(\d+)\.(\d+)$'
    match = re.match(pattern, input_str)
    
    if match:
        # get a, b, c
        a, b, c = map(int, match.groups())
        return f"block{a}.{b-1}"
    else:
        return None



# Function to modify ResNet
@torch.no_grad()
def C_D_resnet(model):
    """
    Replaces all Conv2d and BatchNorm2d layers in the ResNet model
    according to the specified rules.
    """
    for name, module in model.named_modules():
        
        # only prune each conv1 of each block
        if args.resnet_conv1_only:
            if 'conv2' not in name:
                continue
            
        # Decompressor (except the first one, as in_channels=3)
        if isinstance(module, nn.Conv2d) and module.in_channels != 3:
            
            parent_name = ".".join(name.split(".")[:-1])  # Get the parent module's name
            module_name = name.split(".")[-1]  # Get the current module's name
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]
            # args.logger.info('Decompressor parent_name',parent_name, '; module_name', module_name )
            # Replace the Conv2d layer
            setattr(parent_module, module_name, Decompressor(module))
            
        # Decompressor for Linear layer
        elif isinstance(module, torch.nn.Linear):
            assert (name=='fc')
            setattr(model, name, nn.Sequential(nn.Linear(module.in_features,module.in_features, bias=False), module))  
            nn.init.eye_(model.fc[0].weight.data) # identity initialization

    # Compressor
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            
            # only prune each conv1 of each block
            if args.resnet_conv1_only: 
                if name=='bn1' or 'down' in name or 'bn1' not in name:
                    continue
            parent_name = ".".join(name.split(".")[:-1])  # Get the parent module's name
            module_name = name.split(".")[-1]  # Get the current module's name
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]

            # Replace the BatchNorm2d layer
            setattr(parent_module, module_name, Compressor_bcb(module))


@torch.no_grad()
def C_D_modify_model(model, model_name):
    if 'vgg' in model_name:
        C_D_vgg(model)
    elif 'resnet' in model_name:
        C_D_resnet(model)
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")

def get_param_groups(model, pruner, lr, weight_decay):
    one_by_one_conv_params = []
    one_by_one_conv_params_first_layer = []
    other_params = []
    
    for m_name, module in model.named_modules():
        
        if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1)  and module.stride == (1, 1):
            # compressor or decompressor for pruning each conv1 of basicblock
            if args.first_layer_lr_div!=None and (('bn1' in name and not name.startswith('bn1')) or 'conv2' in name) and 'resnet' in args.model:
                for name, param in module.named_parameters(recurse=False):
                    one_by_one_conv_params_first_layer.append(param)
                    args.logger.info(f'one_by_one_conv_params_first_layer: {m_name}.{name}')
            else:        
            
                for name, param in module.named_parameters(recurse=False):
                    one_by_one_conv_params.append(param)
                    args.logger.info(f'one_by_one_conv_params: {m_name}.{name}')
        elif isinstance(module, torch.nn.Linear) and m_name[-2:]=='.0':
            for name, param in module.named_parameters(recurse=False):
                one_by_one_conv_params.append(param)
                args.logger.info(f'one_by_one_fc_params: {m_name}.{name}')
        else:
            for name, param in module.named_parameters(recurse=False):
                other_params.append(param)
                
    if args.first_layer_lr_div==None:
        if one_by_one_conv_params==[]:
            args.logger.info(f'\n one_by_one_conv_params==[]\n')
            return [{"params": other_params, "lr": lr, "weight_decay": weight_decay if pruner is None else 0}]
        else:
            return [
                {"params": other_params, "lr": lr, "weight_decay": weight_decay if pruner is None else 0},
                {"params": one_by_one_conv_params, "lr": lr/args.lr_div, "weight_decay": weight_decay/args.wd_div if args.wd_div!=0 else 0},
            ]
    else:
        return [
                {"params": other_params, "lr": lr, "weight_decay": weight_decay if pruner is None else 0},
                {"params": one_by_one_conv_params, "lr": lr/args.lr_div, "weight_decay": weight_decay/args.wd_div if args.wd_div!=0 else 0},
                {"params": one_by_one_conv_params_first_layer, "lr": lr/args.first_layer_lr_div, "weight_decay": weight_decay/args.wd_div if args.wd_div!=0 else 0},
            ]
        
def progressive_pruning(_pruner, orig_model, speed_up, example_inputs, train_loader=None, EP_model=None):
    orig_model.eval()
    base_ops, base_para = tp.utils.count_ops_and_params(orig_model, example_inputs=example_inputs)
    
    
    # try several time to get the best pruned model w.r.t to pruned accuracy
    best_acc, smallest_loss, best_idxs, select_speed_up, select_para_remain = -1, 1e10, None, 0, 0
    
    repeats = 1
    for repeat in range(repeats):
        
        args.logger.info(f'{"="*20}Repeat {repeat+1}/{repeats} {"="*20}')
        # proxy_model = copy.deepcopy(orig_model) # proxy model
        proxy_model = registry.get_model(args.model, num_classes=args.num_classes, pretrained=True, target_dataset=args.dataset).to(args.device)
        proxy_model.load_state_dict(orig_model.state_dict())
        proxy_model.eval()
        
        pruner = get_pruner(proxy_model, example_inputs=example_inputs, default_depgraph=False) # use pruner decided by args.method 

        current_speed_up = 1
        pruning_record_orig_wo = {}
        while current_speed_up < speed_up:
            proxy_model.eval()
            
            if args.method == "obdc":
                proxy_model.zero_grad()
                imp=pruner.importance
                imp._prepare_model(proxy_model, pruner)
                for k, (imgs, lbls) in enumerate(train_loader):
                    if k>=10: break
                    imgs = imgs.to(args.device)
                    lbls = lbls.to(args.device)
                    output = proxy_model(imgs)
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),
                                                    1).squeeze().to(args.device)
                    loss_sample = F.cross_entropy(output, sampled_y)
                    loss_sample.backward()
                    imp.step()
                # pruner.step()
                imp._rm_hooks(proxy_model)
                imp._clear_buffer()
                
            elif "Jacobian" in args.method or 'jacobian' in args.method:
                proxy_model.zero_grad()
                imp = pruner.importance
                imp.zero_grad()
                imp.zero_score()
                N_batchs = args.N_batchs if args.N_batchs>0 else len(train_loader)
                for k, (imgs, lbls) in enumerate(train_loader):
                    if k>=N_batchs: break
                    imgs = imgs.to(args.device)
                    lbls = lbls.to(args.device)
                    output = proxy_model(imgs)
                    loss = torch.nn.functional.cross_entropy(output, lbls)
                    proxy_model.zero_grad() # clear gradients
                    loss.backward()
                    imp.accumulate_grad(proxy_model) # accumulate Jacobian                       
                    if (k+1)%10==0 or k+1==N_batchs:
                        imp.accumulate_score(proxy_model) # accumulate scores so that the CUDA memory is not exhausted
                # pruner.step()
            # else:
                # pruner.step()
            
            for id, group in enumerate(pruner.step(interactive=True)): 
                dep, deleted_idxs = group[0]
                target_module = dep.target.module
                layer_name = dep.target.name[:dep.target.name.find(' ')] # Get the layer name
                # pruning_fn = dep.handler
                if layer_name not in pruning_record_orig_wo:
                    pruning_record_orig_wo[layer_name] = [np.arange(target_module.out_channels), np.arange(target_module.out_channels)]

                # runing_record_orig_wo[layer_name][0] is the original full index
                # runing_record_orig_wo[layer_name][1] is the remained index; 
                pruning_record_orig_wo[layer_name][1] = np.delete(pruning_record_orig_wo[layer_name][1], deleted_idxs)
                # prune here
                group.prune()
            
            remain_ops, remain_parameters = tp.utils.count_ops_and_params(proxy_model, example_inputs=example_inputs)
            current_speed_up = float(base_ops) / remain_ops
            args.logger.info(f'current_speed_up vs speed_up: {current_speed_up} {speed_up}')
            if pruner.current_step == pruner.iterative_steps:
                break
            
        
        proxy_train_acc, proxy_train_loss = eval_f(proxy_model, train_loader, device=args.device)
        args.logger.info(f'Train Acc {proxy_train_acc}; Loss {proxy_train_loss}; Ops {remain_ops}, {current_speed_up}x; Par {remain_parameters}, {remain_parameters/base_para*100}%; Best Train Acc {best_acc}')
        if best_acc<proxy_train_acc:
            best_acc = proxy_train_acc
            smallest_loss = proxy_train_loss
            best_idxs = pruning_record_orig_wo
            select_speed_up = current_speed_up
            select_para_remain = '{:.2f}%'.format(remain_parameters/base_para*100)

        del pruner
        del proxy_model # remove reference
        # in case CUDA out of memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    args.logger.info(f'Smallest Train loss: {smallest_loss}; Best Train Acc: {best_acc}; Speed up: {select_speed_up}x; Para remain: {select_para_remain}\n')
    

    # pruned_idxs = all_idxs - remained_idxs
    pruning_record_orig_wo = best_idxs
    args.logger.info('\n')
    for item in pruning_record_orig_wo:
        args.logger.info(f'{item}, #Original filter={len(pruning_record_orig_wo[item][0])}, #Remain={len(pruning_record_orig_wo[item][1])}, #Pruned={len(pruning_record_orig_wo[item][0])-len(pruning_record_orig_wo[item][1])}')
        pruning_record_orig_wo[item] = np.setdiff1d(pruning_record_orig_wo[item][0], pruning_record_orig_wo[item][1])
    args.logger.info('\n')
    
    # prune the original model    
    orig_model.eval()
    DG = tp.DependencyGraph().build_dependency(orig_model, example_inputs)
    for name, module in orig_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name in pruning_record_orig_wo:
            idxs = pruning_record_orig_wo[name]
            group = DG.get_pruning_group(module,  tp.prune_conv_out_channels, idxs)
            group.prune()  
            
    # Prune the EP_model according to the original model pruned idxs
    if EP_model!=None:
        
        EP_model.eval()
        DG = tp.DependencyGraph().build_dependency(EP_model, example_inputs)
        for name, module in EP_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
            
                if 'vgg' in args.model:
                    pruned_key = check_and_modify_vgg(name)
                    if not pruned_key:
                        continue       
                elif 'resnet' in args.model:
                    # find the compressor in the 'bn-conv-bn' block
                    if 'downsample.1.1' in name:
                        pruned_key = name[:-3]+'0' # (Compressor) 'xxxx.downsample.1.1'  <-->  'xxxx.downsample.0' (Original Conv layer)
                    elif name[-2:]=='.1' and  'bn' in name: 
                        pruned_key = name.replace('bn', 'conv') # (Compressor)  'xxxx.bnx.1'  <-->  'xxxx.convx' (Original Conv layer)
                        pruned_key = pruned_key[:-2]
                    else:
                        continue
                else:
                    raise NotImplementedError
                        
                if pruned_key in pruning_record_orig_wo:
                    idxs = pruning_record_orig_wo[pruned_key]
                    group = DG.get_pruning_group(module,  tp.prune_conv_out_channels, idxs)
                    group.prune()  
        
        if args.select=='BCB_replace':
            with torch.no_grad():
                EP_model_state_dict = EP_model.state_dict()
                for name, module in EP_model.named_modules():
                    # continue
                    if isinstance(module, torch.nn.BatchNorm2d) and (('down' not in name and name[-2:]=='.0') or ('down' in name and name[-4:]=='.1.0')):
                        args.logger.info(f'reset old bn: {name}') 
                        conv_weight = EP_model_state_dict[name[:-1] + '1.weight'].reshape(EP_model_state_dict[name[:-1] + '1.weight'].shape[:2])
                        # print(conv_weight.shape)
                        # assert(len(conv_weight.shape)==2)

                        indices = torch.argwhere(conv_weight==1)[:,1]
                        # args.logger.info(f'{len(indices)}, {conv_weight.shape}')
                        assert(len(indices)==conv_weight.shape[0])
                        for idx in indices:
                            module.weight.data[idx] = 1.0
                            module.bias.data[idx] = 0.0
                            module.running_var.data[idx] = 1.0
                            module.running_mean.data[idx] = 0.0
                            module.eps = 0
        
        # delete those unpruned Compressor-Decompressor
        for name, module in EP_model.named_modules():
            if (isinstance(module, torch.nn.Conv2d) and module.in_channels==module.out_channels and module.kernel_size==(1,1) and module.stride==(1,1)) \
                or (isinstance(module, torch.nn.Linear) and module.in_features==module.out_features  and name.endswith('.0')):
                    parent_module = EP_model
                    module_names = name.split(".")  
                    for sub_name in module_names[:-1]:
                        parent_module = getattr(parent_module, sub_name)  
                    
                    # replace as Identity
                    setattr(parent_module, module_names[-1], nn.Identity())
                    args.logger.info(f"Replaced {name} with nn.Identity")
                    
                    
                    # remove the new bn of compressor
                    if isinstance(module, torch.nn.Conv2d) and name.endswith('.1') and 'BCB' in args.select:
                              
                        # assert(name[-1]=='1')
                        if args.select=='BCB':
                            name = name[:-1]+'2' # new bn
                        elif args.select=='BCB_replace':
                            name = name[:-1]+'0' # new bn (actually the old bn but assign with new value, while the actual new bn become old bn)
                        parent_module = EP_model
                        module_names = name.split(".")  
                        for sub_name in module_names[:-1]:
                            parent_module = getattr(parent_module, sub_name)  
                        
                        setattr(parent_module, module_names[-1], nn.Identity())
                        args.logger.info(f"Replaced {name} with nn.Identity")                    

                
    del DG 
                  
    return current_speed_up


def eval_f(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            if torch.isnan(out).any():
                input(f'Nan')
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    lr_decay_milestones,
    lr_decay_gamma=0.1,
    save_as=None,
    
    # For pruning
    weight_decay=5e-4,
    save_state_dict_only=True,
    pruner=None,
    device=None,
    orig_model=None,
    words = '',
    warm_up=False,
    warm_up_epochs=10,  
    warm_up_factor=0.1,  
    
):
    writer_words = 'naive' if words=='' else words.replace('_', '')
    writer = SummaryWriter(args.output_dir+f'/{writer_words}')
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if words=='EP_':
        param_groups = get_param_groups(model, pruner, lr, weight_decay)

        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay if pruner is None else 0,
        )
    
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    
    
    if eval(args.warmup):
        args.logger.info(f'{"="*20}Warm-up{"="*20}\n\n') 
        input('warm up right?')   
        # Define warm-up scheduler (LinearLR)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warm_up_factor, total_iters=warm_up_epochs)  # Warm-up for first 10 epochs

        # Combine warm-up and MultiStepLR using SequentialLR
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warm_up_epochs]  # Switch to MultiStepLR after 10 epochs
        )
        epochs += warm_up_epochs
    
    model.to(device)
    best_acc = -1
    
    for epoch in range(epochs):
        epoch_train_loss = []
        model.train()
        

        
        if 'EP' in words and args.freeze_c2:
            if 'resnet' in args.model:
                # if epoch<5: # freeze
                if True:
                    for name, module in model.named_modules():
                        # decompressor or compressor for each conv2 of block
                        if isinstance(module, torch.nn.Conv2d) and module.kernel_size==(1,1) and module.stride==(1,1):
                            
                            if name[:3]=='bn1' or 'down' in name or 'bn2' in name or 'conv1' in name:
                                if epoch==0:
                                    args.logger.info(f'freeze_c2 Conv: {name}') 
                                if hasattr(module, 'weight') and module.weight!=None: # freeze weight
                                    module.weight.requires_grad_(False)
                                if hasattr(module, 'bias') and module.bias!=None: # freeze bias
                                    module.bias.requires_grad_(False) 
                        
                        # new bn for each conv2 of block
                        elif isinstance(module, torch.nn.BatchNorm2d) and (('down' not in name and (name.endswith('.bn2.2') or name=='bn1.2')) or ('down' in name and name.endswith('.1.2'))):
                            if epoch==0:
                                args.logger.info(f'freeze_c2 new bn: {name}') 
                            module.eval() # freeze running_var, running_mean
                            if hasattr(module, 'weight'): # freeze weight
                                module.weight.requires_grad_(False)
                            if hasattr(module, 'bias'): # freeze bias
                                module.bias.requires_grad_(False)     
                else: # unfreeze
                    for name, module in model.named_modules():
                        # decompressor or compressor
                        if isinstance(module, torch.nn.Conv2d) and module.kernel_size==(1,1) and module.stride==(1,1):
                            if epoch==0:
                                args.logger.info(f'freeze_c2 Conv: {name}') 
                            if name[:3]=='bn1' or 'down' in name or 'bn2' in name or 'conv1' in name:
                                if hasattr(module, 'weight')  and module.weight!=None: # freeze weight
                                    module.weight.requires_grad_(True)
                                if hasattr(module, 'bias')  and module.bias!=None: # freeze bias
                                    module.bias.requires_grad_(True) 
                        
                        # new bn
                        elif isinstance(module, torch.nn.BatchNorm2d) and (('down' not in name and name[-2:]=='.2') or ('down' in name and name[-4:]=='.1.2')):
                            if epoch==0:
                                args.logger.info(f'freeze_c2 new bn: {name}') 
                            module.train() # freeze running_var, running_mean
                            if hasattr(module, 'weight'): # freeze weight
                                module.weight.requires_grad_(True)
                            if hasattr(module, 'bias'): # freeze bias
                                module.bias.requires_grad_(True)         
            else:
                assert NotImplementedError

        if 'EP' in words and args.freeze_c1:
            if 'resnet' in args.model:
                # if epoch<5: # freeze
                if True:
                    for name, module in model.named_modules():
                        # decompressor or compressor
                        if isinstance(module, torch.nn.Conv2d) and module.kernel_size==(1,1) and module.stride==(1,1):

                            # each compressor or decompressor for conv1 of each block (except conv1; downsample)
                            if (not name.startswith('bn1') and 'bn1' in name) or 'conv2' in name:
                                if epoch==0:
                                    args.logger.info(f'freeze_c1 Conv: {name}') 
                                if hasattr(module, 'weight') and module.weight!=None: # freeze weight
                                    module.weight.requires_grad_(False)
                                if hasattr(module, 'bias') and module.bias!=None: # freeze bias
                                    module.bias.requires_grad_(False) 
                        
                        # new bn  for conv1 of each block
                        elif isinstance(module, torch.nn.BatchNorm2d) and 'down' not in name and not name.startswith('bn1') and 'bn1.2' in name:
                            if epoch==0:
                                args.logger.info(f'freeze_c1 new bn: {name}') 
                            module.eval() # freeze running_var, running_mean
                            if hasattr(module, 'weight'): # freeze weight
                                module.weight.requires_grad_(False)
                            if hasattr(module, 'bias'): # freeze bias
                                module.bias.requires_grad_(False)     
                else: # unfreeze
                    for name, module in model.named_modules():
                        # decompressor or compressor
                        if isinstance(module, torch.nn.Conv2d) and module.kernel_size==(1,1) and module.stride==(1,1):
                            if epoch==0:
                                args.logger.info(f'freeze_c2 Conv: {name}') 
                            if name[:3]=='bn1' or 'down' in name or 'bn2' in name or 'conv1' in name:
                                if hasattr(module, 'weight')  and module.weight!=None: # freeze weight
                                    module.weight.requires_grad_(True)
                                if hasattr(module, 'bias')  and module.bias!=None: # freeze bias
                                    module.bias.requires_grad_(True) 
                        
                        # new bn
                        elif isinstance(module, torch.nn.BatchNorm2d) and (('down' not in name and name[-2:]=='.2') or ('down' in name and name[-4:]=='.1.2')):
                            if epoch==0:
                                args.logger.info(f'freeze_c2 new bn: {name}') 
                            module.train() # freeze running_var, running_mean
                            if hasattr(module, 'weight'): # freeze weight
                                module.weight.requires_grad_(True)
                            if hasattr(module, 'bias'): # freeze bias
                                module.bias.requires_grad_(True)         
            else:
                assert NotImplementedError



        # if 'EP' in words and (args.freeze_old_bn or args.select=='BCB_replace'):
        if 'EP' in words and args.freeze_old_bn:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d) and (('down' not in name and name.endswith('.0')) or ('down' in name and name.endswith('.1.0'))):
                    if epoch==0:
                        args.logger.info(f'\n{"="*20}freeze old bn: {name}') 
                    module.eval() # freeze running_var, running_mean
                    if hasattr(module, 'weight'): # freeze weight
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'): # freeze bias
                        module.bias.requires_grad_(False)
        # if 'EP' in words:
        #     args.logger.info(f'\n{"="*20}Epoch{epoch} fc[0] weight norm: ')
        #     args.logger.info(torch.norm(model.fc[0].weight.data, dim=1))                        

        for i, (data, target) in enumerate(train_loader):
            # if epoch<1 and i==0:
            #     model.eval()
            # else:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            
            loss = F.cross_entropy(out, target)
            epoch_train_loss.append(loss.item())
            # args.logger.info(f'epoch {epoch}; i {i}')
            
            # for n, m in model.named_modules():
            #     if isinstance(m, torch.nn.BatchNorm2d):
            #         args.logger.info(f'{"="*20}{n}\nweight\n{m.weight}\nbias\n{m.bias}\nrunning_var\n{m.running_var}\nrunning_mean\n{m.running_mean}\n')
            if  torch.any(torch.isnan(out)) or torch.any(torch.isnan(loss)):
                args.logger.info(f'{epoch}-{i} Nan')
                args.logger.info('NAN!')
                args.logger.info(f'out {out}')
                args.logger.info(f'loss {loss}') 
                input('contine?')
                       
            # args.logger.info('out:')
            # args.logger.info(out) 
            
            
            
            # Knowledge distillation
            if 'SL' not in words and args.coeff_label > 0 and orig_model is not None:
                orig_model.eval()
                with torch.no_grad():
                    teacher_out = orig_model(data)
                KD_loss =  F.kl_div(
                    F.log_softmax(out/args.T, dim=1), # Student predictions (softened)
                    F.softmax(teacher_out/args.T, dim=1), # Teacher predictions (softened)
                    reduction='batchmean' # Note that default is 'mean', which is divided by the number of elements in the output
                    ) *   (args.T ** 2) # Scale by temperature squared
                
                # Combine CE loss and KD loss
                loss = args.coeff_ce * loss + args.coeff_label * KD_loss
                
            loss.backward()

            
            if pruner is not None:
                pruner.regularize(model) # for sparsity learning
                
            optimizer.step()
            
            # model.eval()
            # acc, val_loss = eval_f(model, test_loader, device=device)
            # args.logger.info(
            #     "Epoch {:d}-{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
            #         epoch, i, acc, val_loss, optimizer.param_groups[0]["lr"]
            #     )
            # )    
            
            
            if i % 10 == 0 and args.verbose:
                args.logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )
        
        
        if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
            pruner.update_reg() # increase the strength of regularization
            #args.logger.info(pruner.group_reg[pruner._groups[0]])
        
        model.eval()    
        acc, val_loss = eval_f(model, test_loader, device=device)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )
        
        writer.add_scalar(f'train_loss', np.array(epoch_train_loss).mean(), epoch)
        writer.add_scalar(f'val_loss', val_loss, epoch)
        writer.add_scalar(f'val_acc', acc, epoch)
        
        # input('contine?')
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "prune":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}_{}{}_{:.2f}x.pth".format(args.dataset, args.model, words, args.method, 1) )

                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}.pth".format(args.dataset, args.model) )
                torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    args.logger.info("Best Acc=%.4f" % (best_acc))
    check_and_add_row(args.excel_path, args.result_name,  words+'top1', args.save_suffix, acc*100)
    check_and_add_row(args.excel_path, args.result_name,  words+'best_top1', args.save_suffix, best_acc*100)

def get_pruner(model, example_inputs, default_depgraph=False):
    # args.sparsity_learning = False
    
    # use the default manner as DepGraph to regularize the model for sparse learning
    if default_depgraph:
        normalizer_for_sl = None if args.normalizer_for_sl=='None' else args.normalizer_for_sl
        imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer=normalizer_for_sl) # follwing depGraph; normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
        
    else:
        if args.method == "random":
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "l1":
            imp = tp.importance.MagnitudeImportance(p=1, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "l2":
            imp = tp.importance.MagnitudeImportance(p=2, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "fpgm":
            imp = tp.importance.FPGMImportance(p=2, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "obdc":
            imp = tp.importance.OBDCImportance(group_reduction='mean', num_classes=args.num_classes, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "lamp":
            imp = tp.importance.LAMPImportance(p=2, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
        elif args.method == "slim":
            args.sparsity_learning = True
            imp = tp.importance.BNScaleImportance(normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
        elif args.method == "group_slim":
            args.sparsity_learning = True
            imp = tp.importance.BNScaleImportance(normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning, group_lasso=True)
        elif args.method == "group_norm":
            imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
        elif args.method == "group_sl":
            args.sparsity_learning = True
            imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer=args.normalizer) # normalized by the maximum score for CIFAR
            pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
        elif args.method == "growing_reg":
            args.sparsity_learning = True
            imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
        elif args.method == "WHC":
            imp = WHCImportance(normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MetaPruner, global_pruning=args.global_pruning)
        elif args.method == "Jacobian" or 'jaco' in args.method:
            imp = GroupJacobianImportance_accumulate(group_reduction=args.group_reduction, normalizer=args.normalizer)
            pruner_entry = partial(tp.pruner.MetaPruner, global_pruning=args.global_pruning)
        else:
            raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    # ignore output layers
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
            ignored_layers.append(m)
            
        if isinstance(m, torch.nn.Conv2d) and args.resnet_conv1_only and 'resnet' in args.model:
            if name=='conv1' or 'conv2' in name or 'down' in name:
                ignored_layers.append(m)
    
    # Following DepGraph
    # use a small step until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=args.max_pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner


def main():
    global args
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # if args.select=='BCB_replace':
    #     args.freeze_old_bn = True
    
    if args.normalizer=='None':
        args.normalizer = None
    if args.normalizer_for_sl=='None':
        args.normalizer_for_sl = None
    if args.group_reduction=='None':
        args.group_reduction = None   
          
    # if args.sparsity_learning:
    #     if args.normalizer_for_sl=='max':
    #         if args.model=='resnet56':
    #             args.sl_restore = 'results/base_model/CIFAR10/reg_cifar10_resnet56_Jacobian_0.0005.pth'
    #         elif args.model=='vgg19':
    #             args.sl_restore = 'results/base_model/CIFAR10/reg_cifar100_vgg19_Jacobian_0.0005.pth'
    # else:
    #     if args.model=='resnet56':
    #         args.restore = 'results/base_model/DepGraph/cifar10_resnet56.pth'
    #     elif args.model=='vgg19':
    #         args.restore = 'results/base_model/DepGraph/cifar100_vgg19.pth'
    
    middle_name = ''
    if args.resnet_conv1_only:
        middle_name += '_Conv1Only'
    if args.freeze_old_bn:
        middle_name += '_freezeOldBN'
    if args.sparsity_learning:
        middle_name += '_Sparse'
    
    if args.first_layer_lr_div!=None:
        middle_name += f'_FirstLayerLrDiv{args.first_layer_lr_div}' 
        
    args.output_dir = f"{args.output_dir}_{args.select}{middle_name}_GR{args.group_reduction}_SlNor{args.normalizer_for_sl}_lr{args.lr_div}_wd{args.wd_div}_RC{args.randomC}_RD{args.randomD}" 
        
    # args.output_dir = args.output_dir + '_lr'+str(args.lr_div)+'_wd'+str(args.wd_div)
    # input(args.output_dir)
    args, result_name, exp_file, excel_path   = save_dir_name_pruned(args)
    try:
        check_and_add_row(excel_path, 'save_dir', 'Speedup', 0, args.output_dir)
    except Exception as e:
        # Set up logging to write errors to bug.log
        logging.basicConfig(filename="./bug.log", level=logging.ERROR, format="%(asctime)s - %(message)s")

        logging.error(f"Error occurred while processing:\nExcel Path: {excel_path}\nOutput Directory: {args.output_dir}\nError: {str(e)}")
        print(f"An error occurred. Check the log file 'bug.log' for details.")
        print(f"An unexpected error occurred: {e}")
        
        

        
    args.logger = utils.get_logger(None, output=exp_file)

    # Model & Dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes, train_dst, val_dst, input_size = registry.get_dataset(
        args.dataset, data_root=args.dataroot
    )
    args.num_classes = num_classes
    model = registry.get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=args.batch_size,
        num_workers=16, 
        drop_last=False,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=args.batch_size, num_workers=8, shuffle=False
    )
    
    for k, v in utils.utils.flatten_dict(vars(args)).items():  # args.logger.info args
        args.logger.info("%s: %s" % (k, v))

    if args.restore is not None:
        loaded = torch.load(args.restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))
    model = model.to(args.device)


    ######################################################
    # Training / Pruning / Testing
    example_inputs = train_dst[0][0].unsqueeze(0).to(args.device)
    if args.mode == "pretrain":
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        train_model(
            model=model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader,
            words='pretrain_'
        )
        
        
        
    elif args.mode == "prune":
        
        # pruner = get_pruner(model, example_inputs=example_inputs)
        
        ###################  0. Sparsity Learning   ###################  
        if args.sparsity_learning:
            
            if not args.sl_restore:
                reg_pth = "reg_{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.reg)
                reg_pth = os.path.join( os.path.join(args.output_dir, reg_pth) )
                
                # we use the default manner as DepGraph to regularize the model for sparse learning
                pruner = get_pruner(model, example_inputs=example_inputs, default_depgraph=True)
                
                args.logger.info("Regularizing...")
                train_model(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.sl_total_epochs,
                    lr=args.sl_lr,
                    lr_decay_milestones=args.sl_lr_decay_milestones,
                    lr_decay_gamma=args.lr_decay_gamma,
                    pruner=pruner,
                    save_state_dict_only=True,
                    save_as = reg_pth,
                    words = 'SL_'
                )
                del pruner # remove reference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                    
                args.logger.info("Loading the sparse model from {}...".format(reg_pth))
                model.load_state_dict( torch.load(reg_pth, map_location=args.device) )
                
            else:
                args.logger.info("Loading the sparse model from {}...".format(args.sl_restore))
                model.load_state_dict( torch.load(args.sl_restore, map_location=args.device) )
            
        
        
        
        #########################  1. Pruning  ######################### 
        # original_model = copy.deepcopy(model)
        if args.coeff_label>0: # for knowledge-distillation
            original_model = registry.get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset).to(args.device)
            original_model.load_state_dict(model.state_dict())
        
        model.eval()
        ori_ops, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        ori_acc, ori_val_loss = eval_f(model, test_loader, device=args.device)
        args.logger.info(f"\nOriginal Model Val_acc: {ori_acc}, val_loss: {ori_val_loss} ")
        
        # equivalent pruning
        if args.equivalent:
            # EP_model = copy.deepcopy(model)
            EP_model = registry.get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset).to(args.device)
            EP_model.load_state_dict(model.state_dict())
            
            EP_model = EP_model.to(args.device)
            C_D_modify_model(EP_model, args.model)
            EP_model = EP_model.to(args.device)
            EP_model.eval()
            EP_ori_acc, EP_ori_val_loss = eval_f(EP_model, test_loader, device=args.device)
            args.logger.info(f"\nOriginal EP_model Val_acc: {EP_ori_acc}, val_loss: {EP_ori_val_loss} ") 
        else:
            EP_model = None
            
        args.logger.info("Pruning...")
        # No pruner here
        progressive_pruning(None, model, speed_up=args.speed_up, example_inputs=example_inputs, train_loader=train_loader, EP_model=EP_model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        args.logger.info(f'{"="*20}Pruned Model {args.model}{"="*20}')
        args.logger.info(f'{model}\n') 
        if args.equivalent:
            args.logger.info(f'{"="*20}Pruned equivalence Model {args.model}{"="*20} ')
            args.logger.info(f'{EP_model}\n') 
        
        
        args.logger.info(f'{"="*20}Evalutating raw Pruned Model{"="*20}')
        model.eval()
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        train_pruned_acc, train_pruned_val_loss = eval_f(model, train_loader, device=args.device)
        pruned_acc, pruned_val_loss = eval_f(model, test_loader, device=args.device)
        check_and_add_row(excel_path, result_name,  'Speedup', args.save_suffix, ori_ops/pruned_ops)
        check_and_add_row(excel_path, result_name,  'Params_rate', args.save_suffix, pruned_size/ori_size*100)
        check_and_add_row(excel_path, result_name,  'raw1', args.save_suffix, pruned_acc*100)
        args.real_speedup = ori_ops/pruned_ops   
        args.logger.info("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100))
        args.logger.info( "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_ops / 1e6, pruned_ops / 1e6, pruned_ops / ori_ops * 100, ori_ops / pruned_ops,))
        args.logger.info("Val Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
        args.logger.info("Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss))
        args.logger.info("Train Acc: {:.4f}; Train Loss: {:.4f}".format(train_pruned_acc, train_pruned_val_loss))
        
        if EP_model!=None:
            args.logger.info(f'{"="*20}Evalutating raw Equivalent Pruned Model{"="*20}')
            EP_model.eval()
            EP_pruned_acc, EP_pruned_val_loss = eval_f(EP_model, test_loader, device=args.device)
            EP_train_pruned_acc, EP_train_pruned_val_loss = eval_f(EP_model, train_loader, device=args.device)
            args.logger.info("Val Acc: {:.4f} => {:.4f}".format(ori_acc, EP_pruned_acc))
            args.logger.info("Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, EP_pruned_val_loss))
            args.logger.info("Train Acc: {:.4f}; Train Loss: {:.4f}".format(EP_train_pruned_acc, EP_train_pruned_val_loss))
            
        
        # 2. Finetuning
        if args.finetune:
            if args.equivalent:
                args.logger.info(f"\n{'='*20}Finetuning the Equivalent pruned model {'='*20}")
                train_model(
                    EP_model,
                    epochs=args.total_epochs,
                    lr=args.lr,
                    lr_decay_milestones=args.lr_decay_milestones,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=args.device,
                    save_state_dict_only=False,
                    orig_model=original_model if args.coeff_label>0 else None,
                    words='EP_')
                args.logger.info('\n')
                
            args.logger.info(f"\n{'='*20}Finetuning the original pruned model {'='*20}")
            train_model(
                model,
                epochs=args.total_epochs,
                lr=args.lr,
                lr_decay_milestones=args.lr_decay_milestones,
                train_loader=train_loader,
                test_loader=test_loader,
                device=args.device,
                save_state_dict_only=False, 
                orig_model=original_model if args.coeff_label>0 else None)
            
            
    elif args.mode == "test":
        model.eval()
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        acc, val_loss = eval_f(model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc, val_loss))

if __name__ == "__main__":
    main()
