'''
CUDA_VISIBLE_DEVICES=0 python benchmark_ViT.py --data_path /your/data/root --repeats 10 --N_batchs 50 --normalizer None 
CUDA_VISIBLE_DEVICES=1 python benchmark_ViT.py --data_path /your/data/root --repeats 10 --N_batchs 50 --normalizer None --global_pruning
CUDA_VISIBLE_DEVICES=2 python benchmark_ViT.py --data_path /your/data/root --repeats 10 --N_batchs 50 --normalizer None --global_pruning --bottleneck --save_prefix Bottleneck
'''

import copy
import os, sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import torch.nn.functional as F
import torch_pruning as tp
import timm
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode

import presets

import argparse


from Selfmake_Importance import GroupJacobianImportance, WHCImportance, GroupJacobianImportance_accumulate
import numpy as np

# matplotlib与其他包冲突，必须放到最后导入，否则严重影响validate速度；可能是版本问题
from matplotlib import pyplot as plt
from utils import pretrain_path


def parse_args():
    parser = argparse.ArgumentParser(description='Timm ViT Pruning')
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str, help='model name')
    parser.add_argument('--data_path', default='../dataset/ILSVRC2012', type=str, help='model name')
    parser.add_argument('--repeats', type=int, default=10, help='how many times of repeating the experiment')
    parser.add_argument('--save_dir', type=str, default='Transformer', help='Folder to save checkpoints and log.')
    parser.add_argument('--save_prefix', type=str, default='', help='Folder prefix to save checkpoints and log.')
    parser.add_argument('--global_pruning', action='store_true', help='global pruning or local pruning; default is local pruning')
    parser.add_argument('--N_batchs', type=int, default=50, help='how many batchs to use for importance estimation; if -1, use all')
    parser.add_argument('--normalizer', type=str, choices=['parameters','mean','max', 'sum', 'standarization', 'None'], default='None')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--prune_num_heads', default=False, action='store_true', help='global pruning')
    parser.add_argument('--head_pruning_ratio', default=0.0, type=float, help='head pruning ratio')
    parser.add_argument('--use_imagenet_mean_std', default=False, action='store_true', help='use imagenet mean and std')
    parser.add_argument('--val_batch_size', default=64, type=int, help='val batch size') 
    parser.add_argument('--ignore_imps', nargs='*', default=[], type=str, help='ignore importance')
    args = parser.parse_args()
    # args = parser.parse_known_args()

    return args


# Here we re-implement the forward function of timm.models.vision_transformer.Attention
# as the original forward function requires the input and output channels to be identical.
def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=16, use_imagenet_mean_std=False):
    """The imagenet_root should contain train and val folders.
    """

    print('Parsing dataset...')
    train_dst = ImageFolder(os.path.join(imagenet_root, 'train'), 
                            transform=presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=256,
                                interpolation=InterpolationMode.BILINEAR,
                            )
    )
    val_dst = ImageFolder(os.path.join(imagenet_root, 'val'), 
                          transform=presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=256,
                                interpolation=InterpolationMode.BILINEAR,
                            )
    )
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for k, (images, labels) in enumerate(tqdm(val_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)






if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_inputs = torch.randn(1,3,224,224)

    # Load the model
    model = timm.create_model(args.model_name, pretrained=True).eval().to(device)
    input_size = [3, 224, 224]
    example_inputs = torch.randn(1, *input_size).to(device)
        
    train_loader, val_loader = prepare_imagenet(args.data_path, train_batch_size=64, val_batch_size=args.val_batch_size, use_imagenet_mean_std=args.use_imagenet_mean_std)


    N_batchs = args.N_batchs if args.N_batchs!=-1 else len(train_loader)
    global_pruning = args.global_pruning
    normalizer=None if args.normalizer=='None' else args.normalizer
    # save_prefix = '0'
    # save_prefix = ''
    repeats = args.repeats



    ###
    # Importance criteria
    imp_dict = {
        # data-free
        'Group WHC': WHCImportance(group_reduction='sum', normalizer=normalizer),  
        'Group L1': tp.importance.MagnitudeImportance(p=1, group_reduction='sum', normalizer=normalizer),
        'Group FPGM': tp.importance.FPGMImportance(group_reduction='sum', normalizer=normalizer),
        'Random': tp.importance.RandomImportance(),        
        
        # data-driven
        # 'Group Jacobian': GroupJacobianImportance(group_reduction='sum', normalizer=normalizer),  
        'Group Jacobian': GroupJacobianImportance_accumulate(group_reduction='sum', normalizer=normalizer),  
        'Group Taylor': tp.importance.TaylorImportance(group_reduction='sum', normalizer=normalizer),
        'Group Hessian': tp.importance.HessianImportance(group_reduction='sum', normalizer=normalizer),  # Hessian is too slow
    }


    colors = {
        'Group WHC': 'C0',         # Blue
        'Group L1': 'C1',          # Orange
        'Group FPGM': 'C2',        # Green
        'BNScaleImportance': 'C7', # Gray
        'Random': 'C4',            # Purple
        
        'Group Taylor': 'C5',      # Brown
        'Group Hessian': 'C6',     # Pink
        'Group Jacobian': 'C3',    # red
    }


    params_record = {}
    train_loss_record = {}
    train_acc_record = {}
    val_loss_record = {}
    val_acc_record = {}
    macs_record = {}



    print(f'Validating the original model {args.model_name}...')
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    base_train_acc, base_train_loss = validate_model(model, train_loader, device) # tak too long time
    # base_train_acc, base_train_loss = 0, 0
    base_val_acc, base_val_loss = validate_model(model, val_loader, device)
    # base_val_acc, base_val_loss = 0, 0 
    print(f"MACs: {base_macs/base_macs:.2f}, #Params: {base_nparams/base_nparams:.2f}, Train_Acc: {base_train_acc:.4f}, train_Loss: {base_train_loss:.4f}, Val_Acc: {base_val_acc:.4f}, Val_Loss: {base_val_loss:.4f}")

    for imp_name, imp in imp_dict.items():
        if imp_name in args.ignore_imps:
            continue
        for repeat in range(repeats):
            
            # deteminted criteria do no need multiple test
            if not ('Taylor' in imp_name or 'Hessian' in imp_name or 'Jacobian' in imp_name or 'Random' in imp_name) and repeat>=1:
                continue
            
            print(f"{'='*50}{imp_name} {repeat+1}/{repeats}{'='*50}")
            
            if imp_name not in params_record:
                train_loss_record[imp_name] = [[]]
                train_acc_record[imp_name] = [[]]
                val_loss_record[imp_name] = [[]]
                val_acc_record[imp_name] = [[]]        
                params_record[imp_name] = [[]]
                macs_record[imp_name] = [[]]
            else:
                train_loss_record[imp_name].append([])
                train_acc_record[imp_name].append([])
                val_loss_record[imp_name].append([])
                val_acc_record[imp_name].append([])       
                params_record[imp_name].append([])
                macs_record[imp_name].append([])
            
            params_record[imp_name][repeat].append(base_nparams)
            train_loss_record[imp_name][repeat].append(base_train_loss)
            train_acc_record[imp_name][repeat].append(base_train_acc)
            val_loss_record[imp_name][repeat].append(base_val_loss)
            val_acc_record[imp_name][repeat].append(base_val_acc)    
            macs_record[imp_name][repeat].append(base_macs)  
                        
            
            model = timm.create_model(args.model_name, pretrained=True).eval().to(device)
            model.eval()
            
            # Note that even variable `model` is rewritten, the cuda memory is not released
            # So we need to release the memory manually  using empty_cache() to avoid out-of-memory
            torch.cuda.empty_cache()
            
            num_heads = {}
            ignored_layers = [model.head]
            for m in model.modules():
                if isinstance(m, timm.models.vision_transformer.Attention):
                    m.forward = forward.__get__(m, timm.models.vision_transformer.Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
                    num_heads[m.qkv] = m.num_heads 
                if args.bottleneck and isinstance(m, timm.models.vision_transformer.Mlp): 
                    ignored_layers.append(m.fc2) # only prune the internal layers of FFN & Attention

            iterative_steps = 10
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs,
                iterative_steps=iterative_steps,
                importance=imp,
                pruning_ratio=0.5, 
                ignored_layers=ignored_layers,
                global_pruning=global_pruning,
                num_heads=num_heads, # number of heads in self attention
                prune_num_heads=args.prune_num_heads, # reduce num_heads by pruning entire heads (default: False)
                prune_head_dims=not args.prune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
                head_pruning_ratio=args.head_pruning_ratio, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
                round_to=1)

            for i in range(iterative_steps):
                model.eval()
                
                if isinstance(imp, tp.importance.HessianImportance):
                    imp.zero_grad() # clear accumulated gradients before each pruning step
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs) 
                        # compute loss for each sample
                        loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                        for l in loss:
                            model.zero_grad() # clear gradients
                            l.backward(retain_graph=True) # simgle-sample gradient
                            imp.accumulate_grad(model) # accumulate g^2       
                        
                        torch.cuda.empty_cache() # in case of CUDA out-of-memory
                            
                elif isinstance(imp, tp.importance.TaylorImportance):
                    model.zero_grad() # clear accumulated gradients before each pruning step
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs)
                        loss = torch.nn.functional.cross_entropy(output, lbls)
                        loss.backward() 
                                                    
                elif isinstance(imp, GroupJacobianImportance):
                    imp.zero_grad() # clear accumulated gradients
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs) 
                        loss = torch.nn.functional.cross_entropy(output, lbls)
                        model.zero_grad() # clear gradients
                        loss.backward()
                        imp.accumulate_grad(model) # accumulate Jacobian   

                elif isinstance(imp, GroupJacobianImportance_accumulate):
                    imp.zero_grad() # clear accumulated gradients
                    imp.zero_score() # clear accumulated scores
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs) 
                        loss = torch.nn.functional.cross_entropy(output, lbls)
                        model.zero_grad() # clear gradients
                        loss.backward()
                        imp.accumulate_grad(model) # accumulate Jacobian                       
                        if (k+1)%10==0 or k+1==N_batchs:
                            imp.accumulate_score(model) # accumulate scores so that the CUDA memory is not exhausted
                        
                                            
                pruner.step()
                # Modify the attention head size and all head size aftering pruning
                head_id = 0
                for m in model.modules():
                    if isinstance(m, timm.models.vision_transformer.Attention):
                        # print("Head #%d"%head_id)
                        # print("[Before Pruning] Num Heads: %d, Head Dim: %d =>"%(m.num_heads, m.head_dim))
                        m.num_heads = pruner.num_heads[m.qkv]
                        m.head_dim = m.qkv.out_features // (3 * m.num_heads)
                        # print("[After Pruning] Num Heads: %d, Head Dim: %d"%(m.num_heads, m.head_dim))
                        # print()
                        head_id+=1                
                # print(model)
                
                model.eval()
                macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
                val_acc, val_loss = validate_model(model, val_loader, device)
                # train_acc, train_loss = validate_model(model, train_loader, device) # tak too long time
                # val_acc, val_loss = 0, 0 
                train_acc, train_loss = 0, 0
                print(f"{imp_name} {repeat+1}/{repeats} MACs: {macs/base_macs:.2f}, #Params: {nparams/base_nparams:.2f}, Train_Acc: {train_acc:.4f}, Train_Loss: {train_loss:.4f}, Val_Acc: {val_acc:.4f}, Val_Loss: {val_loss:.4f}")
                params_record[imp_name][repeat].append(nparams)
                train_loss_record[imp_name][repeat].append(train_loss)
                train_acc_record[imp_name][repeat].append(train_acc)
                val_loss_record[imp_name][repeat].append(val_loss)
                val_acc_record[imp_name][repeat].append(val_acc)
                macs_record[imp_name][repeat].append(macs)



        ######################### Draw and save  when each importance criteria is finished  #########################
        ######################### Just to avoid missing all the data when things happen such of CUDA out-of-memory  
        # for fig in ['train', 'val']:
        for fig in ['Validate']:
            if fig == 'Train':
                acc_record, loss_record = train_acc_record, train_loss_record
            elif fig == 'Validate':
                acc_record, loss_record = val_acc_record, val_loss_record
                
            # Parameters vs Accuracy
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                # plt.plot(params_record[imp_name][0], np.mean(acc_record[imp_name], axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(params_record[imp_name][0], np.mean(acc_record[imp_name], axis=0), yerr=np.std(acc_record[imp_name], axis=0), fmt='o', ms=4, capsize=4, color=colors[imp_name], linestyle='-', label=imp_name)
            plt.xlabel('# Parameters')
            plt.ylabel(f'{fig} Accuracy')
            plt.legend(fontsize='small', loc='best')
            os.makedirs(f'./results/benchmark_importance/{args.save_dir}/{args.model_name}', exist_ok=True)
            plt.savefig(f'./results/benchmark_importance/{args.save_dir}/{args.model_name}/{args.save_prefix}Par_{fig}Acc_Bottle{args.bottleneck}_Ph{args.prune_num_heads}_Hr{args.head_pruning_ratio}_Nbatchs{N_batchs}_normalizer{normalizer}_global{global_pruning}_repeats{args.repeats}.pdf')


            # Parameters vs Loss
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                # plt.plot(params_record[imp_name][0], np.mean(loss_record[imp_name], axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(params_record[imp_name][0], np.mean(loss_record[imp_name], axis=0), yerr=np.std(loss_record[imp_name], axis=0), fmt='o', ms=4, capsize=4, color=colors[imp_name], linestyle='-', label=imp_name)
            plt.xlabel('# Parameters')
            plt.ylabel(f'{fig} Loss')
            plt.legend(fontsize='small', loc='best')
            plt.savefig(f'./results/benchmark_importance/{args.save_dir}/{args.model_name}/{args.save_prefix}Par_{fig}Loss_Bottle{args.bottleneck}_Ph{args.prune_num_heads}_Hr{args.head_pruning_ratio}_Nbatchs{N_batchs}_normalizer{normalizer}_global{global_pruning}_repeats{args.repeats}.pdf')


            # Macs vs Accuracy
            plt.figure()
            for index, imp_name in enumerate(macs_record.keys()):
                # plt.plot(macs_record[imp_name][0], np.mean(acc_record[imp_name], axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(macs_record[imp_name][0], np.mean(acc_record[imp_name], axis=0), yerr=np.std(acc_record[imp_name], axis=0), fmt='o', ms=4, capsize=4, color=colors[imp_name], linestyle='-', label=imp_name)     
            plt.xlabel('MACs')
            plt.ylabel(f'{fig} Accuracy')
            plt.legend(fontsize='small', loc='best')
            plt.savefig(f'./results/benchmark_importance/{args.save_dir}/{args.model_name}/{args.save_prefix}Mac_{fig}Acc_Bottle{args.bottleneck}_Ph{args.prune_num_heads}_Hr{args.head_pruning_ratio}_Nbatchs{N_batchs}_normalizer{normalizer}_global{global_pruning}_repeats{args.repeats}.pdf')

            # Macs vs Loss
            plt.figure()
            for index, imp_name in enumerate(macs_record.keys()):
                # plt.plot(macs_record[imp_name][0], np.mean(loss_record[imp_name],axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(macs_record[imp_name][0], np.mean(loss_record[imp_name], axis=0), yerr=np.std(loss_record[imp_name], axis=0), fmt='o', ms=4, capsize=4, color=colors[imp_name], linestyle='-', label=imp_name)
                
            plt.xlabel('MACs')
            plt.ylabel(f'{fig} Loss')
            plt.legend(fontsize='small', loc='best')
            plt.savefig(f'./results/benchmark_importance/{args.save_dir}/{args.model_name}/{args.save_prefix}Mac_{fig}Loss_Bottle{args.bottleneck}_Ph{args.prune_num_heads}_Hr{args.head_pruning_ratio}_Nbatchs{N_batchs}_normalizer{normalizer}_global{global_pruning}_repeats{args.repeats}.pdf')

        # save the results when each importance criteria is finished
        torch.save({'params_record':params_record, 'macs_record':macs_record, \
            'train_loss_record':train_loss_record, 'train_acc_record':train_acc_record, \
            'val_acc_record':val_acc_record, 'val_loss_record':val_loss_record},\
            f'./results/benchmark_importance/{args.save_dir}/{args.model_name}/{args.save_prefix}Bottle{args.bottleneck}_Ph{args.prune_num_heads}_Hr{args.head_pruning_ratio}_Nbatchs{N_batchs}_normalizer{normalizer}_global{global_pruning}_repeats{args.repeats}.pth')
        


