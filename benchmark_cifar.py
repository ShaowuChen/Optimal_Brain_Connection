
'''
pip install torch-pruning --upgrade

CUDA_VISIBLE_DEVICES=0 python benchmark_cifar.py --model vgg19 --pth_path ./results/base_model/cifar100_vgg19.pth --data_root /your/data/root --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.9 --iterative_steps 18 

CUDA_VISIBLE_DEVICES=1 python benchmark_cifar.py --model resnet56 --pth_path ./results/base_model/cifar10_resnet56.pth --data_root /your/data/root --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.9 --iterative_steps 18 

'''

import logging
import sys, os
import time
import registry

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
import presets
from torchvision.transforms.functional import InterpolationMode


import torch_pruning as tp
import os
from tqdm import tqdm

from Selfmake_Importance import GroupJacobianImportance_accumulate, GroupJacobianImportance, WHCImportance
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='pruning')

parser.add_argument('--model', metavar='ARCH',  default='vgg19',help='path to dataset')
parser.add_argument('--save_dir', type=str, default='', help='Folder to save checkpoints and log.')

parser.add_argument('--run_criteria', type=str, default='', help='select which criteria for running; if '', run all; if ["L1"] for example, run "L1" only')

parser.add_argument('--global_pruning', action='store_true', help='global pruning or local pruning; default is local pruning')
parser.add_argument('--N_batchs', type=int, default=-1, help='how many batchs to use for importance estimation; if -1, use all')
parser.add_argument('--group_reduction', type=str, choices=['mean', 'first', 'max'], default='sum')
parser.add_argument('--normalizer', type=str, choices=['mean','max', 'sum', 'standarization', 'None'], default='None')

parser.add_argument('--repeats', type=int, default=5, help='how many times for repeat experiments')
parser.add_argument('--pruning_ratio', type=float, default=0.9)
parser.add_argument('--iterative_steps', type=int, default=18)

parser.add_argument('--data_root', type=str, default='',required=True)
parser.add_argument('--pth_path', type=str, default='',required=True)


args = parser.parse_args()


def validate_model(model, val_loader):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)



if __name__ == "__main__":

    if 'vgg19' in args.model:
        dataset = 'cifar100'    
        num_classes = 100
        pth_path = args.pth_path
    elif 'resnet56' in args.model:
        dataset = 'cifar10'
        num_classes = 10
        pth_path = args.pth_path
    else:
        raise NotImplementedError
    
    ####### load data
    if dataset in ['cifar10', 'cifar100']:
        batch_size = 128
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        train_transform = transforms.Compose(
            [transforms.ToTensor(),  transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        if dataset=='cifar10':
            train_data = datasets.CIFAR10(args.data_root, train=True, transform=train_transform, download=True)
            test_data = datasets.CIFAR10(args.data_root, train=False, transform=test_transform, download=True)        
        elif dataset=='cifar100':
            train_data = datasets.CIFAR100(args.data_root, train=True, transform=train_transform, download=True)
            test_data = datasets.CIFAR100(args.data_root, train=False, transform=test_transform, download=True)         
        else:
            raise NotImplementedError
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=8, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=8, persistent_workers=True)    
    
    else:
        raise NotImplementedError

        
        
    ###### prepare hyper-parameters
    N_batchs = args.N_batchs if args.N_batchs!=-1 else len(train_loader)
    global_pruning = args.global_pruning
    network = args.model
    normalizer=None if args.normalizer=='None' else args.normalizer
    group_reduction=None  if args.group_reduction=='None' else args.group_reduction
    
    repeats = args.repeats
    pruning_ratio = args.pruning_ratio
    iterative_steps = args.iterative_steps # remove 5% filter each time by default
    
    save_dir = f'./results/benchmark_importance/{args.save_dir}/{args.model}_{dataset}_Rate{pruning_ratio}_IterSteps{iterative_steps}_Repeat{repeats}_NBatch{N_batchs}_BatchSize{batch_size}'
    if global_pruning:
        save_dir += '_Global'
    else:
        save_dir += '_Local'        
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger(save_dir+'/log.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s- %(message)s',
                        handlers=[
                            logging.FileHandler(save_dir+'/log.log'),
                            logging.StreamHandler()
                        ])

    logger = logging.getLogger()

    ### Importance criteria
    imp_dict = {
        # # data-free
        'Random': tp.importance.RandomImportance(), 
        'L1': tp.importance.GroupMagnitudeImportance(p=1, group_reduction=group_reduction, normalizer=normalizer),
        'FPGM': tp.importance.FPGMImportance(group_reduction=group_reduction, normalizer=normalizer),
        'BN Scale': tp.importance.BNScaleImportance(group_reduction=group_reduction, normalizer=normalizer),
        'WHC': WHCImportance(group_reduction=group_reduction, normalizer=normalizer),      
        
        # # data-driven
        'Taylor': tp.importance.TaylorImportance(group_reduction=group_reduction, normalizer=normalizer),  
        'Jacobian': GroupJacobianImportance_accumulate(group_reduction=group_reduction, normalizer=normalizer), 
        'Hessian': tp.importance.HessianImportance(group_reduction=group_reduction, normalizer=normalizer),
    }
    
    
    colors = {
        'WHC': 'C0',         # Blue
        'L1': 'C1',          # Orange
        'FPGM': 'C2',        # Green
        'BN Scale': 'C7', # Gray
        'Random': 'C4',            # Purple
        
        'Taylor': 'C5',      # Brown
        'Hessian': 'C6',     # Pink
        'Jacobian': 'C3',    # red
        'Jacobian Isolated': 'black',    # black
    }


    time_record = {}
    params_record = {}
    macs_record = {}
    train_loss_record = {}
    train_acc_record = {}
    val_loss_record = {}
    val_acc_record = {}
    

    pretrained=False
    example_inputs = torch.randn(1, 3, 32, 32).cuda()
        
    model = registry.get_model(args.model, num_classes=num_classes, pretrained=pretrained, target_dataset=dataset).cuda()
    if not pretrained:
        state_dict = torch.load(pth_path, weights_only=False)
        model.load_state_dict(state_dict)    
    model.eval()
    
    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    base_train_acc, base_train_loss = 0, 0
    base_val_acc, base_val_loss = validate_model(model, val_loader) 
    logger.info(f"MACs: {base_macs/base_macs:.2f}, #Params: {base_nparams/base_nparams:.2f}, Train_Acc: {base_train_acc:.4f}, train_Loss: {base_train_loss:.4f}, Val_Acc: {base_val_acc:.4f}, Val_Loss: {base_val_loss:.4f}")

    middle_name = f'Gr{group_reduction}_No{normalizer}_Nbatchs{N_batchs}'
    if global_pruning:
        middle_name += '_Global'
    else:
        middle_name += '_Local'
    
    for imp_name, imp in imp_dict.items():
        if args.run_criteria=='':
            pass
        else:
            if imp_name not in eval(args.run_criteria):
                continue
        
        evaluating_time = 0
        overall_start_time = time.time()
        for repeat in range(repeats):
            
            
            # deteminted criteria do no need multiple test
            if not ('Taylor' in imp_name or 'Hessian' in imp_name or 'Jacobian' in imp_name or 'Random' in imp_name) and repeat>=1:
                continue
            
            logger.info('='*50+imp_name+' repeat'+str(repeat)+'='*50)
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
                
            model = registry.get_model(args.model, num_classes=num_classes, pretrained=pretrained, target_dataset=dataset).cuda()
            if not pretrained:
                state_dict = torch.load(pth_path, weights_only=False)
                model.load_state_dict(state_dict)    
            model.eval()
            torch.cuda.empty_cache() 
                        
            example_inputs = torch.randn(1, 3, 32, 32).cuda()
            if 'resnet' in args.model:
                ignored_layers = [model.fc]  # DO NOT prune the final classifier!
            else:
                ignored_layers = [model.classifier]
            
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs,
                iterative_steps=iterative_steps,
                importance=imp,
                pruning_ratio=pruning_ratio, 
                ignored_layers=ignored_layers,
                global_pruning=global_pruning,
            )

            logger.info(f"MACs: {base_macs/base_macs:.2f}, #Params: {base_nparams/base_nparams:.2f}, Train_Acc: {base_train_acc:.4f}, train_Loss: {base_train_loss:.4f}, Val_Acc: {base_val_acc:.4f}, Val_Loss: {base_val_loss:.4f}")

            params_record[imp_name][repeat].append(base_nparams)
            train_loss_record[imp_name][repeat].append(base_train_loss)
            train_acc_record[imp_name][repeat].append(base_train_acc)
            val_loss_record[imp_name][repeat].append(base_val_loss)
            val_acc_record[imp_name][repeat].append(base_val_acc)    
            macs_record[imp_name][repeat].append(base_macs)

            for i in range(iterative_steps):
                model.eval()
                
                evaluating_start_time = time.time()
                if isinstance(imp, tp.importance.HessianImportance):
                    imp.zero_grad() # clear accumulated gradients before each pruning step
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs) 
                        # compute loss for each sample
                        loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                        '''
                        Note, the code from torch_pruning wrongly clear the gradients of the model here
                        so we remove the following line 
                        '''
                        # imp.zero_grad() # clear accumulated gradients
                        for l in loss:
                            model.zero_grad() # clear gradients
                            l.backward(retain_graph=True) # simgle-sample gradient
                            imp.accumulate_grad(model) # accumulate g^2  
                        torch.cuda.empty_cache() # in case CUDA OUT OF MEMORY                    
                            
                elif isinstance(imp, tp.importance.TaylorImportance):
                    model.zero_grad() # clear accumulated gradients before each pruning step
                    for k, (imgs, lbls) in enumerate(train_loader):
                        if k>=N_batchs: break
                        imgs = imgs.cuda()
                        lbls = lbls.cuda()
                        output = model(imgs)
                        loss = torch.nn.functional.cross_entropy(output, lbls)
                        loss.backward() 
                    torch.cuda.empty_cache()  # in case CUDA OUT OF MEMORY
                                                    
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
                    torch.cuda.empty_cache()  # in case CUDA OUT OF MEMORY  
                
                elif isinstance(imp, GroupJacobianImportance_accumulate):
                    imp.zero_score()
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
                        if (k+1)%50==0 or k==N_batchs-1:
                            imp.accumulate_score(model)
                            torch.cuda.empty_cache()  # in case CUDA OUT OF MEMORY
                            
                pruner.step()
                evaluating_time += time.time()-evaluating_start_time

                model.eval()
                macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
                val_acc, val_loss = validate_model(model, val_loader)
                if dataset!='imagenet':
                    train_acc, train_loss = validate_model(model, train_loader)
                else:
                    train_acc, train_loss = 0, 0
                logger.info(f"MACs: {macs/base_macs:.2f}, #Params: {nparams/base_nparams:.2f}, Train_Acc: {train_acc:.4f}, Train_Loss: {train_loss:.4f}, Val_Acc: {val_acc:.4f}, Val_Loss: {val_loss:.4f}")
                params_record[imp_name][repeat].append(nparams)
                train_loss_record[imp_name][repeat].append(train_loss)
                train_acc_record[imp_name][repeat].append(train_acc)
                val_loss_record[imp_name][repeat].append(val_loss)
                val_acc_record[imp_name][repeat].append(val_acc)
                macs_record[imp_name][repeat].append(macs)

            torch.cuda.empty_cache()
        
        time_record[imp_name] = [(time.time() - overall_start_time)/repeats/iterative_steps, evaluating_time/repeats/iterative_steps]
        logger.info(f'{imp_name} average overall time (including evaluation): {time_record[imp_name][0]}; average evaluating time: {time_record[imp_name][1]}')
        
        

        ######################### save #########################
        torch.save({'iterative_steps': iterative_steps, 'pruning_ratio':pruning_ratio, 'N_batchs':N_batchs, 'batch_size':batch_size, \
                    'params_record':params_record, 'macs_record':macs_record, 'train_loss_record':train_loss_record,\
                    'train_acc_record':train_acc_record, 'val_acc_record':val_acc_record, 'val_loss_record':val_loss_record},\
                    f'{save_dir}/a_record_{network}_{middle_name}.pth')

       
        ######################### draw #########################
        
        
        figs = ['Train', 'Validate'] if dataset!='imagenet' else ['Validate']
        for fig in figs:
            if fig == 'Train':
                acc_record, loss_record = train_acc_record, train_loss_record
            elif fig == 'Validate':
                acc_record, loss_record = val_acc_record, val_loss_record

            # Pruned propotion vs Accuracy
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                plt.errorbar(np.linspace(0, pruning_ratio, iterative_steps+1)*100, np.mean(acc_record[imp_name], axis=0), yerr=np.std(acc_record[imp_name], axis=0), fmt='o', ms=4, capsize=3, color=colors[imp_name], linestyle='-', label=imp_name)
            plt.xlabel('Pruned Filters (%)')
            plt.ylabel(f'{fig} Accuracy')
            plt.legend(fontsize='xx-small', loc='best')
            plt.savefig(f'{save_dir}/Propotion_{fig}_acc_{middle_name}.pdf')
            plt.close()


            # Pruned propotion vs Loss
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                plt.errorbar(np.linspace(0, pruning_ratio, iterative_steps+1), np.mean(loss_record[imp_name], axis=0), yerr=np.std(loss_record[imp_name], axis=0), fmt='o', ms=4, capsize=3, color=colors[imp_name], linestyle='-', label=imp_name)
            plt.xlabel('Pruned Filters (%)')
            plt.ylabel(f'{fig} Loss')
            plt.legend(fontsize='xx-small', loc='best')
            plt.savefig(f'{save_dir}/Propotion_{fig}_loss_{middle_name}.pdf')
            plt.close()


                
            # Parameters vs Accuracy
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                # plt.plot(params_record[imp_name][0], np.mean(acc_record[imp_name], axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(np.array(params_record[imp_name]).mean(axis=0), np.mean(acc_record[imp_name], axis=0), yerr=np.std(acc_record[imp_name], axis=0), fmt='o', ms=4, capsize=3, color=colors[imp_name], linestyle='-', label=imp_name)
            plt.xlabel('# Parameters')
            plt.ylabel(f'{fig} Accuracy')
            plt.legend(fontsize='xx-small', loc='best')
            plt.savefig(f'{save_dir}/params_{fig}_acc_{middle_name}.pdf')
            plt.close()


            # Parameters vs Loss
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                # plt.plot(params_record[imp_name][0], np.mean(loss_record[imp_name], axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(np.array(params_record[imp_name]).mean(axis=0), np.mean(loss_record[imp_name], axis=0), yerr=np.std(loss_record[imp_name], axis=0), fmt='o', ms=4, capsize=3, color=colors[imp_name], linestyle='-', label=imp_name)
            plt.xlabel('# Parameters')
            plt.ylabel(f'{fig} Loss')
            plt.legend(fontsize='xx-small', loc='best')
            plt.savefig(f'{save_dir}/params_{fig}_loss_{middle_name}.pdf')
            plt.close()


            # Macs vs Accuracy
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                # follow the same rule
                # plt.plot(macs_record[imp_name][0], np.mean(acc_record[imp_name], axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(np.array(macs_record[imp_name]).mean(axis=0), np.mean(acc_record[imp_name], axis=0), yerr=np.std(acc_record[imp_name], axis=0), fmt='o', ms=4, capsize=3, color=colors[imp_name], linestyle='-', label=imp_name)     
            plt.xlabel('# MACs')
            plt.ylabel(f'{fig} Accuracy')
            plt.legend(fontsize='xx-small', loc='best')
            plt.savefig(f'{save_dir}/macs_{fig}_acc_{middle_name}.pdf')
            plt.close()

            # Macs vs Loss
            plt.figure()
            for index, imp_name in enumerate(params_record.keys()):
                # plt.plot(macs_record[imp_name][0], np.mean(loss_record[imp_name],axis=0), label=imp_name, color=colors[imp_name])
                plt.errorbar(np.array(macs_record[imp_name]).mean(axis=0), np.mean(loss_record[imp_name], axis=0), yerr=np.std(loss_record[imp_name], axis=0), fmt='o', ms=4, capsize=3, color=colors[imp_name], linestyle='-', label=imp_name)
                
            plt.xlabel('# MACs')
            plt.ylabel(f'{fig} Loss')
            plt.legend(fontsize='xx-small', loc='best')
            plt.savefig(f'{save_dir}/macs_{fig}_loss_{middle_name}.pdf')
            plt.close()





        ############ compare the time
        imp_names = list(time_record.keys())
        overall_times = [time_record[imp_name][0] for imp_name in imp_names]
        evaluating_times = [time_record[imp_name][1] for imp_name in imp_names]

        # Average Pruning Time
        bar_width = 0.45
        # plt.figure(figsize=(10, 6)) 
        plt.figure()
        bar1 = plt.bar(np.arange(0,len(imp_names))/2, overall_times, width=bar_width, label=imp_names, color=[colors[imp_name] for imp_name in imp_names])
        for bar in bar1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                    ha='center', va='bottom', rotation=45)
        plt.xticks(np.arange(0,len(imp_names))/2, imp_names, rotation=15, ha='right')
        plt.xlabel('Criteria')
        plt.ylabel('Average Pruning Time (s)')
        plt.tight_layout()
        plt.legend(fontsize='small', loc='best')
        plt.savefig(f'{save_dir}/Overall_Time.pdf')
        plt.close()

        # Average Evaluating Time
        # plt.figure(figsize=(10, 6)) 
        plt.figure()
        bar2 = plt.bar(range(len(imp_names)), evaluating_times, width=bar_width, label=imp_names, color=[colors[imp_name] for imp_name in imp_names])
        for bar in bar2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                    ha='center', va='bottom', rotation=45)
        plt.xticks(range(len(imp_names)), imp_names, rotation=15, ha='right')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Criteria')
        plt.ylabel('Average Evaluating Time (s)')
        plt.tight_layout()
        plt.legend(fontsize='small', loc='best')
        plt.savefig(f'{save_dir}/Evaluating_Time.pdf')
        # plt.show()
        plt.close()

        



        


