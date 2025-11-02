import shutil
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

import torch
import logging
from torchvision import datasets, transforms
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
# from filelock import FileLock

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter


def check_and_add_row(file_path, row_index, column_index, top1, top5, key_word='top', MACs_Params=None):
    # lock_path = file_path + ".lock"  # 创建锁文件
    # with FileLock(lock_path):
    # Check if file exists
    if os.path.exists(file_path):
        # Read the excel file
        df = pd.read_excel(file_path, index_col=0, engine='openpyxl')
    else:
        # Create a DataFrame with specified columns if the file does not exist
        df = pd.DataFrame(columns=[
            'MACs', 'Params', 'MACs_rate', 'Params_rate',
            'raw1_0', 'raw1_1', 'raw1_2', 'raw1_AVE', 
            'top1_0', 'top1_1', 'top1_2', 'top1_AVE', 
            'top5_0', 'top5_1', 'top5_2', 'top5_AVE'
        ])
        
    if type(MACs_Params)==list:
        MACs_orig, Params_orig = MACs_Params[0]
        MACs_decom, Params_decom = MACs_Params[1]

        df.loc[row_index[0], 'MACs'] = MACs_orig
        df.loc[row_index[0], 'Params'] = Params_orig

        df.loc[row_index[1], 'MACs'] = MACs_decom
        df.loc[row_index[1], 'Params'] = Params_decom   
            
        df.loc[row_index[1], 'MACs_rate'] = round(MACs_decom / MACs_orig * 100, 2)
        df.loc[row_index[1], 'Params_rate'] = round(Params_decom / Params_orig * 100, 2)
        
    else:
        # Update the specified cell
        df.loc[row_index, key_word + '1_' + column_index] = top1
        if 'top' in key_word or 'raw' in key_word:
            df.loc[row_index, key_word+'5_' + column_index] = top5

    # Calculate averages for row_index '2' only if the first three cells are not NaN
    if column_index == '2':
        # Check and calculate the average for 'raw1' if all three cells have values
        raw1_values = df.loc[row_index, ['raw1_0', 'raw1_1', 'raw1_2']].dropna().astype(float)
        if len(raw1_values) == 3:
            df.loc[row_index, 'raw1_AVE'] = raw1_values.mean().round(2)
        
        # Check and calculate the average for 'top1' if all three cells have values
        top1_values = df.loc[row_index, ['top1_0', 'top1_1', 'top1_2']].dropna().astype(float)
        if len(top1_values) == 3:
            df.loc[row_index, 'top1_AVE'] = top1_values.mean().round(2)
        
        # Check and calculate the average for 'top5' if all three cells have values
        top5_values = df.loc[row_index, ['top5_0', 'top5_1', 'top5_2']].dropna().astype(float)
        if len(top5_values) == 3:
            df.loc[row_index, 'top5_AVE'] = top5_values.mean().round(2)
    
    # Save the DataFrame back to the excel file
    df.to_excel(file_path)

def dataset(args):
    kwargs = {"num_workers": args.num_workers, "pin_memory": args.pin_memory}
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size  # make sure the number of val samples is divisible by (the_number_of_GPUs*val_batch_size)

    if args.data_set=='CIFAR10':
        num_classes = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        
        if args.model[:9]=='MobileNet':
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, padding=4),
                transforms.Resize(256),
                transforms.RandomCrop(args.input_size),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                transforms.Normalize(mean, std)])

            test_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        elif args.model[:6]=='resnet':
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])

        '''
        Note: Here I use cp -r /home/csw163/dataset_sda1/cifar-10-batches-py /dev/shm/cifar-10-batches-py to speed up the data loading
        So you have to copy your CIFAR10 dataset to /dev/shm/ or you can change the data_root to your own path
        '''
        # train_data = datasets.CIFAR10(args.data_root, train=True, transform=train_transform, download=False)
        # test_data = datasets.CIFAR10(args.data_root, train=False, transform=test_transform, download=False)
        
        # 定义源路径和目标路径
        source_dir = args.data_root+"/cifar-10-batches-py"
        target_dir = "/dev/shm/cifar-10-batches-py"
        if not os.path.exists(target_dir):
            try:
                shutil.copytree(source_dir, target_dir)
            except Exception as e:
                print(f"error happen when copy source_dir to target_dir: {e}")
        train_data = datasets.CIFAR10('/dev/shm', train=True, transform=train_transform, download=False)
        test_data = datasets.CIFAR10('/dev/shm', train=False, transform=test_transform, download=False)        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                    num_workers=kwargs['num_workers'], pin_memory=kwargs['pin_memory'], prefetch_factor=4, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False,
                                                    num_workers=kwargs['num_workers'], pin_memory=kwargs['pin_memory'],  prefetch_factor=4, persistent_workers=True)    

    elif args.data_set=='ImageNet':  
        num_classes = 1000  
        train_dataset = datasets.ImageFolder(
                args.data_root + '/ILSVRC2012/train',
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]))          

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=kwargs['num_workers'], pin_memory=kwargs['pin_memory'],
                                                prefetch_factor=8, persistent_workers=True)

        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    args.data_root + '/ILSVRC2012/val', 
                    transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])),            
                batch_size=val_batch_size, shuffle=False,num_workers=kwargs['num_workers'], pin_memory=kwargs['pin_memory'],
                prefetch_factor=8, persistent_workers=True)

    return num_classes, train_loader, val_loader



from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

def get_balanced_subset_loader(args, subset_size, batch_size=128, num_workers=32, pin_memory=True):
    """
    Create class-balanced Subset DataLoader for ImageNet
    Args:
        args: args object
        subset_size (int): subset_size
        batch_size (int): batch_size 
        num_workers (int): DataLoader number of worker
        pin_memory (bool):  pin_memory
    Returns:
        DataLoader: 子集 DataLoader
    """

    train_dataset = datasets.ImageFolder(
                args.data_root + '/ILSVRC2012/train',
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]))          
    
    # 获取类别标签
    targets = [sample[1] for sample in train_dataset.imgs]  # ImageFolder 的 targets 是 `imgs` 属性的第二项
    class_counts = Counter(targets)  # 每个类别的样本数量
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}  # 权重 = 1/样本数量
    sample_weights = [class_weights[target] for target in targets]  # 每个样本的权重

    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=subset_size, replacement=False)

    # 创建 DataLoader
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )





def num_train_examples_per_epoch(dataset_name='CIFAR10'):
    if 'ImageNet' in dataset_name:
        return 1281167
    elif dataset_name in ['CIFAR10']:
        return 50000
    elif dataset_name == 'mnist':
        return 60000
    else:
        assert False


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def sgd_optimizer(model, lr=0.1, weight_decay=1e-4, momentum=0.9, no_l2_keywords=[], use_nesterov=False, keyword_to_lr_mult=None, weight_decay_bias=0):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        if "bias" in key or "bn" in key or "BN" in key:
            apply_weight_decay = weight_decay_bias
            # engine.echo('set weight_decay_bias={} for {}'.format(weight_decay, key))
        else:
            apply_weight_decay = weight_decay

        for kw in no_l2_keywords:
            if kw in key:
                apply_weight_decay = 0
                # engine.echo('NOTICE! weight decay = 0 for {} because {} in {}'.format(key, kw, key))
                break

        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr

        if keyword_to_lr_mult is not None:
            for keyword, mult in keyword_to_lr_mult.items():
                if keyword in key:
                    apply_lr *= mult
                    # engine.echo('multiply lr of {} by {}'.format(key, mult))
                    break

        params += [{"params": [value], "lr": apply_lr, "weight_decay": apply_weight_decay}]
    # optimizer = torch.optim.Adam(params, lr)
    optimizer = torch.optim.SGD(params, lr, momentum=momentum, nesterov=use_nesterov)
    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append((correct_k.item() * 100.0) / batch_size)
    return res



def validate_wo_accelerator(model, valloader, logger, criterion=torch.nn.CrossEntropyLoss(reduction='mean'), info='', info_bool=True):
    """
    Validate a model, compute top-1 and top-5 accuracy, and calculate the average loss.
    
    Args:
        model (torch.nn.Module): The model to validate.
        valloader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): Loss function used to compute the loss.
        
    Returns:
        tuple: (top1_accuracy, top5_accuracy, average_loss)
    """
    model.eval()  # Set the model to evaluation mode
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation for validation
        for i, (inputs, targets) in enumerate(tqdm(valloader)):
            inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
                       
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Accumulate loss
            total_loss += loss.item() * targets.size(0)

            # Get the top-1 and top-5 predictions
            _, pred_top5 = outputs.topk(5, dim=1)
            pred_top1 = pred_top5[:, 0]

            # Update the total number of samples
            total_samples += targets.size(0)

            # Calculate top-1 accuracy
            correct_top1 += (pred_top1 == targets).sum().item()

            # Calculate top-5 accuracy
            correct_top5 += (targets.view(-1, 1) == pred_top5).sum().item()

    top1_accuracy = correct_top1 / total_samples
    top5_accuracy = correct_top5 / total_samples
    average_loss = total_loss / total_samples

    if info_bool:
        logger.info(f"{info} Accuracy: top1: {top1_accuracy:.2f}% top5: {top5_accuracy:.2f}% loss: {average_loss:.4f}")
        
    return top1_accuracy, top5_accuracy, average_loss


def validate(net, accelerator, val_loader, logger, criterion=torch.nn.CrossEntropyLoss(reduction='mean'), info='', info_bool=True):
    '''
    Note that accelerator feeds each GPU the same numbers of samples
    so if the dataset is not divisible by the number of batch size or gpu,
    it may drop some batch or get extra samples from the next loop (depending on the 'drop_last' option of the DataLoader)
    
    Therefore, for the test/validation set, MAKE SURE the number of samples is divisible by number_GPUs*batch_size
    '''
    
    net.eval()
    with torch.no_grad():
        total_samples = 0
        total_top1, total_top5, average_loss = 0.0, 0.0, 0.0
        
        for i, (images, labels) in enumerate(tqdm(val_loader, disable=not accelerator.is_local_main_process)):
            outputs = net(images)

            # Calculate batch validation loss
            b_v_loss = criterion(outputs, labels)
            
            # Calculate accuracy for this batch on each GPU
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            batch_size = torch.tensor(labels.size(0)).to(accelerator.device)
            
            # Gather all metrics into a single variable
            gathered_metrics = accelerator.gather_for_metrics((prec1*batch_size, prec5*batch_size, b_v_loss, batch_size))

            # if accelerator.is_local_main_process:
            total_top1 += gathered_metrics[0].sum().item()
            total_top5 += gathered_metrics[1].sum().item()
            average_loss += gathered_metrics[2].mean().item()
            total_samples += gathered_metrics[3].sum().item()
            
        # Compute final averaged metrics on the main process  
        total_top1 /= total_samples
        total_top5 /= total_samples
        average_loss /= (i+1)  # Here i+1 is num_batch per GPU;   num_samples = num_GPU * num_batch per GPU * batch_size per GPU
        
        if accelerator.is_local_main_process and info_bool:
            logger.info(f"{info} Accuracy: top1: {total_top1:.2f}% top5: {total_top5:.2f}% loss: {average_loss:.4f}\n")
            
        return total_top1, total_top5, average_loss


def train_only_distill(args, orig_net, prune_net, accelerator, optimizer, scheduler, train_loader, val_loader, criterion, num_epoch, recorder, logger, result_name, save=True, val_period=1,feature_prune=None, feature_orig=None):
    assert(len(recorder)==6)
    train_top1, train_top5, train_loss, val_top1, val_top5, val_loss = recorder

    # tensorboard
    if accelerator.is_local_main_process:
        writer = SummaryWriter('./'+args.save_dir)
        writer.add_text("save_dir", args.save_dir, 0)


    orig_net.eval()

    # coef_logit = args.coeff_ce
    #5-epoch warmup, initial value of 0.1 and cosine annealing for 100 epochs. 
    for epoch in range(num_epoch):

        # prune_net.train()
        prune_net.eval()
        start_time = time.time()
        
        epoch_losses=[]
        total=0
        top1, top5 = 0, 0
        if accelerator.is_local_main_process:
                print('epoch',epoch, '  ', scheduler.get_last_lr())        
        for i, (x, y) in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):

            prune_output = prune_net(x)

            with torch.no_grad():
                orig_output = orig_net(x)

            kd_label_loss = nn.KLDivLoss()(F.log_softmax(prune_output, dim=1), F.softmax(orig_output, dim=1)) 
            loss1 = kd_label_loss
            
            optimizer.zero_grad()
            accelerator.backward(loss1)
            optimizer.step()
            scheduler.step()
            
            if epoch==0 or (epoch+1)%val_period==0 or epoch==num_epoch-1:
                accelerator.wait_for_everyone()
                loss1, output, y = accelerator.gather_for_metrics((loss1, prune_output, y))
                prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
                top1 += prec1
                top5 += prec5
                loss_record = loss1.mean().item()
                epoch_losses.append(loss_record)
                total += y.size(0)
        
        end_time = time.time()
        # evaluate
        if epoch==0 or (epoch+1)%val_period==0 or epoch==num_epoch-1:
            train_loss.append(np.mean(epoch_losses))
            top1 /= (i+1)
            top5 /= (i+1)
            train_top1.append(top1)
            train_top5.append(top5)

            if accelerator.is_local_main_process:
                logger.info("Epoch {} loss: {} T1_Accuracy: {}%  T5_Accuracy: {}%   Time costs: {}s".format(epoch, train_loss[-1], top1, top5, end_time - start_time))
                writer.add_scalar('Loss/train', train_loss[-1], epoch)
                writer.add_scalar('Top1/train', top1, epoch)
                writer.add_scalar('Top5/train', top5, epoch)
        else:
            if accelerator.is_local_main_process:
                logger.info("Epoch {} last minibatch CE loss: {} Time costs: {}s per epoch".format(epoch, loss1, end_time - start_time))
                       

        if epoch==0 or (epoch+1)%val_period==0 or epoch==num_epoch-1:
            vtop1, vtop5, vloss = validate(prune_net, accelerator, val_loader, logger, info_bool=True)
            val_top1.append(vtop1)
            val_top5.append(vtop5)
            val_loss.append(vloss)
            
            if accelerator.is_local_main_process:
                writer.add_scalar('Loss/val', vloss, epoch)
                writer.add_scalar('Top1/val', vtop1, epoch)
                writer.add_scalar('Top5/val', vtop5, epoch)

        if save and ((epoch+1)%val_period==0 or epoch==num_epoch-1) and accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()            
            state = {'epoch': epoch + 1, 'state_dict': prune_net.state_dict(),
                     'model': accelerator.unwrap_model(prune_net), 
                    'optimizer': optimizer.state_dict(), 'args': args.__dict__,
                    'train_top1_list': train_top1,'train_top5_list': train_top5,'train_loss_list': train_loss,
                    'final_train_top1': top1, 'final_train_top1': top5,
                    'val_top1_list': val_top1,'val_top5_list': val_top5,'val_loss_list': val_loss,
                    'final_val_top1': vtop1, 'final_val_top1': val_top5}
            # accelerator.save(state, './'+args.save_dir+'/'+result_name+'_cp.pth.tar')
            torch.save(state, './'+args.save_dir+'/distill_'+result_name+'_cp.pth.tar')

    return prune_net, None, (train_top1, train_top5, train_loss, val_top1, val_top5, val_loss)



def train(args, orig_net, prune_net, accelerator, optimizer, scheduler, train_loader, val_loader, criterion, num_epoch, recorder, logger, result_name, save=True, val_period=1,feature_prune=None, feature_orig=None):
    assert(len(recorder)==6)
    train_top1, train_top5, train_loss, val_top1, val_top5, val_loss = recorder

    # tensorboard
    if accelerator.is_local_main_process:
        writer = SummaryWriter('./'+args.save_dir)
        writer.add_text("save_dir", args.save_dir, 0)


    orig_net.eval()

    # coef_logit = args.coeff_ce
    #5-epoch warmup, initial value of 0.1 and cosine annealing for 100 epochs. 
    for epoch in range(num_epoch):

        prune_net.train()
        start_time = time.time()
        
        epoch_losses=[]
        total=0
        top1, top5 = 0, 0
        if accelerator.is_local_main_process:
                print('epoch',epoch, '  ', scheduler.get_last_lr())        
        for i, (x, y) in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):

        # for i, (x, y) in enumerate(train_loader):

            prune_output = prune_net(x)
            # loss1 = criterion(prune_output, y)

            # assert(args.coeff_feature+args.coeff_label!=0)
            if args.coeff_label==0 and args.coeff_feature==0:
                loss = criterion(prune_output, y) # ONLY CE loss
            elif args.coeff_label!=0 and args.coeff_feature==0:
                with torch.no_grad():
                    orig_output = orig_net(x)
                kd_label_loss = compute_distillation_label_loss(args, prune_output, orig_output)
                loss = args.coeff_ce * criterion(prune_output, y) + kd_label_loss
            elif args.coeff_label!=0 and args.coeff_feature!=0:
                assert(0) # we do not use feature distillation in this case
                with torch.no_grad():
                    orig_output = orig_net(x)
                kd_feature_loss = compute_distillation_feature_loss(args, feature_prune, feature_orig)
                kd_label_loss = compute_distillation_label_loss(args, prune_output, orig_output)
                loss = args.coeff_ce * criterion(prune_output, y) + kd_feature_loss + kd_label_loss
            else:
                assert(0)

            optimizer.zero_grad()
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()
            scheduler.step()
            # lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            
            
            if epoch==0 or (epoch+1)%val_period==0 or epoch==num_epoch-1:
                accelerator.wait_for_everyone()
                loss, output, y = accelerator.gather_for_metrics((loss, prune_output, y))
                prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
                top1 += prec1
                top5 += prec5
                loss_record = loss.mean().item()
                epoch_losses.append(loss_record)
                total += y.size(0)
        
        end_time = time.time()
        # evaluate
        if epoch==0 or (epoch+1)%val_period==0 or epoch==num_epoch-1:
            train_loss.append(np.mean(epoch_losses))
            top1 /= (i+1)
            top5 /= (i+1)
            train_top1.append(top1)
            train_top5.append(top5)

            if accelerator.is_local_main_process:
                logger.info("Epoch {} loss: {} T1_Accuracy: {}%  T5_Accuracy: {}%   Time costs: {}s".format(epoch, train_loss[-1], top1, top5, end_time - start_time))
                writer.add_scalar('Loss/train', train_loss[-1], epoch)
                writer.add_scalar('Top1/train', top1, epoch)
                writer.add_scalar('Top5/train', top5, epoch)
        else:
            if accelerator.is_local_main_process:
                logger.info("Epoch {} last minibatch loss: {} Time costs: {}s per epoch".format(epoch, loss, end_time - start_time))
                       

        if epoch==0 or (epoch+1)%val_period==0 or epoch==num_epoch-1:
            vtop1, vtop5, vloss = validate(prune_net, accelerator, val_loader, logger, info_bool=True)
            val_top1.append(vtop1)
            val_top5.append(vtop5)
            val_loss.append(vloss)
            
            if accelerator.is_local_main_process:
                writer.add_scalar('Loss/val', vloss, epoch)
                writer.add_scalar('Top1/val', vtop1, epoch)
                writer.add_scalar('Top5/val', vtop5, epoch)


        # save checkpoint
        if save and ((epoch+1)%val_period==0 or epoch==num_epoch-1) and accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()            
            state = {'epoch': epoch + 1, 'state_dict': prune_net.state_dict(),
                     'model': accelerator.unwrap_model(prune_net), 
                    'optimizer': optimizer.state_dict(), 'args': args.__dict__,
                    'train_top1_list': train_top1,'train_top5_list': train_top5,'train_loss_list': train_loss,
                    'final_train_top1': top1, 'final_train_top1': top5,
                    'val_top1_list': val_top1,'val_top5_list': val_top5,'val_loss_list': val_loss,
                    'final_val_top1': vtop1, 'final_val_top1': val_top5}
            # accelerator.save(state, './'+args.save_dir+'/'+result_name+'_cp.pth.tar')
            torch.save(state, './'+args.save_dir+'/'+result_name+'_cp.pth.tar')

    return prune_net, None, (train_top1, train_top5, train_loss, val_top1, val_top5, val_loss)

def compute_distillation_feature_loss(args, s_f, t_f):

    feature_loss = 0

    # loss_func = nn.MSELoss(reduction="mean")
    loss_func = nn.SmoothL1Loss()

    for item in s_f:
        feature_loss = feature_loss + loss_func(s_f[item], t_f[item])

    # ft = torch.cuda.FloatTensor if s_f[item].is_cuda else torch.Tensor

    # feature_loss *= args.coeff_feature/len(s_f)
    feature_loss *= args.coeff_feature

    return feature_loss


def compute_distillation_label_loss(args, s_o, t_o):

    KD_loss = nn.KLDivLoss()(F.log_softmax(s_o/args.T, dim=1),
                             F.softmax(t_o/args.T, dim=1)) * (args.coeff_label * args.T * args.T)

    return KD_loss



# def pretrain_path(model_name, path):
#     if path==None:
#         if 'resnet32' in model_name:
#             pretrain_path = './results/base_model/CIFAR10/resnet32/orig/orig_resnet32_CIFAR10_240EPOCH_no_0_DownsampleC_notinit_correct_optimizer_cp.pth.tar'
#         elif 'resnet56' in model_name:
#             pretrain_path = './results/base_model/CIFAR10/resnet56/orig/orig_resnet56_CIFAR10_240EPOCH_no_2_DownsampleC_notinit_correct_optimizer_cp.pth.tar'
#     else:
#         pretrain_path = path

#     return pretrain_path

def pretrain_path(model_name, path=None):
    if path==None:
        if model_name in ['resnet20','resnet32', 'resnet56']:
            pretrain_path = f'./results/base_model/CIFAR10/FPGM_base/{model_name}checkpoint.pth.tar'
            # resnet32/orig/orig_resnet32_CIFAR10_240EPOCH_no_0_DownsampleC_notinit_correct_optimizer_cp.pth.tar'
        elif model_name in ['resnet20l','resnet32l', 'resnet56l']:
            pretrain_path = f'./results/base_model/CIFAR10/{model_name[:-1]}/orig/orig_{model_name[:-1]}_CIFAR10_240EPOCH_no_0_DownsampleC_notinit_correct_optimizer_cp.pth.tar'
        else:
            raise ValueError(f'No pretrain path for model {model_name}')
    else:
        pretrain_path = path
    return pretrain_path

def silent_conduct(accelerator, f_str):
    if accelerator.is_local_main_process:
        eval(f_str)
    else:
        # 在非主进程上禁用标准输出
        original_stdout = sys.stdout
        sys.stdout = None
        eval(f_str)
        sys.stdout = original_stdout  # 恢复标准输出


def save_dir_name(args, keyword='propose'):
    global_pruning = 'global' if args.global_pruning==True else 'local'
    args.save_dir = './results/'+keyword+'/'+args.save_dir+'/'+args.model+'_'+args.criterion+'_'+args.data_set+'/'+global_pruning+'_'+args.normalizer+'_'+args.ignore_layer_func+'_'+(args.select).replace('_','')+'/'+str(args.expand_rate)+'_div'+str(args.initi_div)+'_eIn'+str(args.expand_input)+'_comConvb'+str(args.add_compressor_convb)+'_comDown'+str(args.add_compressor_down)+'_deDown'+str(args.decompose_downsample)+'/'+str(args.target_flops_reduce_rate)+'_'+str(args.pruning_ratio) 

    # input(global_pruning+'\n'+args.save_dir)


    # save_dir = args.save_dir+'_propose' if args.save_dir!='./' else 'propose'

    # args.save_dir = './results/'+save_dir+'_'+args.criterion+'_'+ args.data_set+'_div'+str(args.initi_div) + '/' +args.model + '/' + args.select +'_'+str(args.expand_rate)+'/'+str(args.target_flops_reduce_rate)+'_'+str(args.pruning_ratio)  
    # global_pruning = 'global' if args.global_pruning==True else 'local'

    # args.save_dir = './results/'+save_dir+'_'+args.criterion+'_'+ args.data_set + '/' +args.model + '/' +global_pruning+'_'+args.normalizer+"_"+args.ignore_layer_func+'/'+ args.select +'_'+str(args.expand_rate)+'_div'+str(args.initi_div)+'_ei'+str(args.expand_input)+'/'+str(args.target_flops_reduce_rate)+'_'+str(args.pruning_ratio) 

    os.makedirs(args.save_dir, exist_ok=True)

    result_name =  args.criterion+'_'+keyword+'_'+args.model+'_'+args.data_set+'_'+global_pruning+'_'+args.ignore_layer_func+'_'+(args.select).replace('_','')+'_er'+str(args.expand_rate)+'_div'+str(args.initi_div)+'_eIn'+str(args.expand_input)+'_comConvb'+str(args.add_compressor_convb)+'_comDown'+str(args.add_compressor_down)+'_deDown'+str(args.decompose_downsample)+'_target'+str(args.target_flops_reduce_rate)+'_pr'+str(args.pruning_ratio) +'_'+'l'+str(args.coeff_label)+'T'+str(args.T)+'f'+str(args.coeff_feature)+'c'+str(args.coeff_ce)+'_epo'+str(args.num_epoch)+'_'+str(args.save_suffix)

    # result_name = args.select + str(args.expand_rate)+'_' + args.model +'_'+args.data_set +'_'+str(args.num_epoch)+'EPOCH_'+str(args.pruning_ratio)+'pr_'+str(args.target_flops_reduce_rate)+'targetRR'+'_'+'l'+str(args.coeff_label)+'T'+str(args.T)+'f'+str(args.coeff_feature)+'c'+str(args.coeff_ce)+'_'+str(args.save_suffix)
    exp_file = './'+args.save_dir+'/'+result_name+'.log'

    excel_path = args.save_dir+'/../../'+args.model+'_results_'+args.data_set+'.xlsx'

    return args, result_name, exp_file, excel_path


def save_dir_name_decom(args):
    conv_names_dir = '_'.join([item.replace('_', '') for item in eval(args.conv_names)])
    # args.save_dir = f'./results/{args.data_set}/decom/{args.save_dir}/{args.model}/{args.decom_method}+{conv_names_dir}/{args.delete_rate}_{args.random_init}_{args.initi_div}_{args.coeff_label}_{args.T}_{args.coeff_feature}_{args.coeff_ce}/{args.save_suffix}'
    args.save_dir = (
    f'./results/{args.data_set}/decom/{args.save_dir}/'
    f'{args.model}/{args.decom_method}+{conv_names_dir}/'
    f'r{args.delete_rate}_{args.random_init}_d{args.initi_div}_'
    f'l{args.coeff_label}_T{args.T}_f{args.coeff_feature}_c{args.coeff_ce}/'
    f'{args.save_suffix}')
        
    os.makedirs(args.save_dir, exist_ok=True)

    result_name = (
        f'{args.model}_{args.decom_method}_{conv_names_dir}_'
        f'r{args.delete_rate}_{args.random_init}_d{args.initi_div}_'
        f'l{args.coeff_label}_T{args.T}_f{args.coeff_feature}_c{args.coeff_ce}_s{args.save_suffix}'
    )
                                    
    exp_file = f'./{args.save_dir}/{result_name}.log'

    # excel_path = f'{args.save_dir}/../../{args.model}_results_{args.data_set}_{args.save_suffix}.xlsx'
    excel_path = f'{args.save_dir}/{args.model}_results_{args.data_set}_{args.save_suffix}.xlsx'
    return args, result_name, exp_file, excel_path

def save_dir_name_pruned(args):
    global_way = 'global' if args.global_pruning==True else 'local'

    args.save_dir = (
    f'./results/{args.data_set}/pruning/{args.save_dir}/{args.tuning_way}/'
    f'{args.model}/{args.importance_criterion}_{args.method}_{global_way}/GR{args.group_reduction}_No{args.normalizer}_NB{args.N_batchs}/'
    f'fr{args.target_flops_reduce_rate}_dr{args.delete_rate}_er{args.expand_rate}_id{args.initi_div}_'
    f'l{args.coeff_label}_T{args.T}_f{args.coeff_feature}_c{args.coeff_ce}/'
    f'{args.save_suffix}')
        
    os.makedirs(args.save_dir, exist_ok=True)

    result_name = (
        f'{args.tuning_way}_{args.model}_{args.importance_criterion}_{args.method}_{global_way}_GR{args.group_reduction}_No{args.normalizer}_NB{args.N_batchs}_'
        f'fr{args.target_flops_reduce_rate}_dr{args.delete_rate}_er{args.expand_rate}_id{args.initi_div}_'
        f'l{args.coeff_label}_T{args.T}_f{args.coeff_feature}_c{args.coeff_ce}_s{args.save_suffix}')
                                        
    exp_file = f'./{args.save_dir}/{result_name}_used_p_idxs{args.used_pruned_idxs}.log'

    # excel_path = f'{args.save_dir}/../../{args.model}_results_{args.data_set}_{args.save_suffix}.xlsx'
    excel_path = f'{args.save_dir}/{args.tuning_way}_{args.model}_{args.importance_criterion}_{args.method}_{global_way}_GR{args.group_reduction}_No{args.normalizer}_{args.save_suffix}.xlsx'
    
    return args, result_name, exp_file, excel_path  

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  
  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
    

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap