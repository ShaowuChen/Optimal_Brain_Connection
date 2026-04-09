import os
import sys
import subprocess
import urllib.request
import json

# [Crucial] Must set the GPU before importing torch!
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

# ==========================================
# 1. Install missing packages automatically
# ==========================================
packages_to_install = []

# HQ-CLIP depends on open_clip_torch and huggingface_hub for loading
try:
    import open_clip
except ImportError:
    packages_to_install.append("open_clip_torch")

try:
    import huggingface_hub
except ImportError:
    packages_to_install.append("huggingface_hub")

try:
    import torch_pruning as tp
except ImportError:
    packages_to_install.append("torch-pruning")

try:
    import scipy
except ImportError:
    packages_to_install.append("scipy")
    
try:
    import torchvision
except ImportError:
    packages_to_install.append("torchvision")
    
try:
    import matplotlib
except ImportError:
    packages_to_install.append("matplotlib")

if packages_to_install:
    print(f"Installing missing packages: {packages_to_install}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages_to_install])

# Download the specific criterion file from the official repo if not present
# if not os.path.exists("Selfmake_Importance.py"):
#     print("Downloading Selfmake_Importance.py from the official repository...")
#     url = "https://raw.githubusercontent.com/ShaowuChen/Optimal_Brain_Connection/main/Selfmake_Importance.py"
#     urllib.request.urlretrieve(url, "Selfmake_Importance.py")

# Download standard ImageNet class mapping if not present
if not os.path.exists("imagenet_class_index.json"):
    print("Downloading standard ImageNet class mapping...")
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    urllib.request.urlretrieve(url, "imagenet_class_index.json")
  

import open_clip
import torch_pruning as tp
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')  
from Selfmake_Importance import GroupJacobianImportance_accumulate, WHCImportance

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}, Current CUDA Device: {torch.cuda.current_device()}")

# The save path is changed to hqclip for differentiation
save_dir = "../hqclip/results/ViT/ImageNet/Both_Brach0"
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 2. HQ-CLIP Model Parameter Definition
# ==========================================
# Referencing the official HQ-CLIP library and model names
MODEL_NAME = "hf-hub:zhixiangwei/vlm150m-hqclip-large-vitb16"
SHORT_NAME = "HQ-CLIP-ViT-B16"  # Used for chart titles
BATCH_SIZE_TEST = 64
BATCH_SIZE_TRAIN = 64
DATA_ROOT = "/data/csw/dataset/ILSVRC2012"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

print("Loading HQ-CLIP preprocessing...")
# open_clip returns: model, preprocess_train, preprocess_val. We need val preprocessing here.
_, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, device="cpu") 
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

print("Preparing ImageNet-1k dataset...")
# Using ImageFolder to load ImageNet data
test_dataset = ImageFolder(root=VAL_DIR, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=4)

train_dataset = ImageFolder(root=TRAIN_DIR, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=4)

# Map ImageFolder classes (which are WordNet IDs like 'n01440764') to human-readable names
with open("imagenet_class_index.json", "r") as f:
    idx2label = json.load(f)
wnid_to_name = {v[0]: v[1].replace('_', ' ') for k, v in idx2label.items()}

# Create a list of human-readable class names aligned with the dataset's class index
classes = [wnid_to_name.get(wnid, wnid) for wnid in test_dataset.classes]

# Use open_clip's tokenizer, supporting list batch processing
text_inputs = tokenizer([f"a photo of a {c}" for c in classes]).to(device)

def evaluate(eval_model, dataloader):
    """Zero-shot Evaluation on ImageNet-1k (HQ-CLIP Adaptation)"""
    eval_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # OpenCLIP requires extracting features first and manually calculating similarity logits
        text_features = eval_model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = eval_model.logit_scale.exp()
        
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            image_features = eval_model.encode_image(imgs)
            image_features = F.normalize(image_features, dim=-1)
            
            # Manually calculate logits
            logits_per_image = logit_scale * image_features @ text_features.T
            
            _, predicted = logits_per_image.max(1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
            
            # Optional: Print progress for large datasets like ImageNet
            if total % 5000 == 0:
                print(f"Eval Progress: {total} images processed...")
                
    return correct / total    

example_image = torch.randn(1, 3, 224, 224).to(device)
example_text = tokenizer(["a photo of a cat"]).to(device)
example_inputs = (example_image, example_text)

def get_model_and_pruner(importance_criterion, target_sparsity, pruning_steps):
    """Reload the full model and build a new Pruner for each experiment"""
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, device=device)
    model = model.float() # Prevent gradient overflow/underflow in half precision
    model.eval()
    
    # Ensure all parameters require gradients to prevent silent failures
    for p in model.parameters():
        p.requires_grad_(True)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            ignored_layers.append(m)

    # Flexible ignore rule configuration for OpenCLIP structure
    if hasattr(model, 'visual'):
        if hasattr(model.visual, 'attnpool'):
            ignored_layers.append(model.visual.attnpool)
        if hasattr(model.visual, 'ln_post'):
            ignored_layers.append(model.visual.ln_post)

    if hasattr(model, 'ln_final'):
        ignored_layers.append(model.ln_final)
    elif hasattr(model, 'text') and hasattr(model.text, 'ln_final'):
        ignored_layers.append(model.text.ln_final)

    unwrapped_parameters = []
    if hasattr(model, 'text_projection') and model.text_projection is not None:
        unwrapped_parameters.append((model, 'text_projection'))
    elif hasattr(model, 'text') and hasattr(model.text, 'proj') and model.text.proj is not None:
        unwrapped_parameters.append((model.text, 'proj'))
        
    if hasattr(model, 'visual') and hasattr(model.visual, 'proj') and model.visual.proj is not None:
        unwrapped_parameters.append((model.visual, 'proj'))

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=importance_criterion,
        global_pruning=True,     
        ch_sparsity=target_sparsity,
        iterative_steps=pruning_steps,  
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters
    )
    return model, pruner

def count_parameters(model):
    """Calculate the total parameters of the current model to verify pruning"""
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, nparams

print("Evaluating Baseline Model...")
base_model, _ = get_model_and_pruner(tp.importance.MagnitudeImportance(p=2), 0.5, 10)
orig_acc = evaluate(base_model, test_loader)
orig_macs, orig_params = count_parameters(base_model)
print(f"--> Baseline Accuracy ({SHORT_NAME}): {orig_acc * 100:.2f}%, Params: {orig_params/1e6:.2f}M")


target_sparsity = 0.9
pruning_steps = 18     # 18 pruning steps
num_batches_for_grad = 50

criteria_names = ['L1', 'FPGM', 'WHC']
# criteria_names = ['L1', 'FPGM', 'WHC', 'Random', 'Taylor', 'Jacobian']

# Result dictionaries, including 0% baseline accuracy and parameter counts
results_acc = {name: [orig_acc] for name in criteria_names}
results_macs = {name: [orig_macs] for name in criteria_names}
results_params = {name: [orig_params] for name in criteria_names}

for crit_name in criteria_names:
    print(f"\n{'='*50}\nStarting Experiments for Criterion: {crit_name}\n{'='*50}")
    
    # 1. Initialize importance criteria
    if crit_name == 'L1':
        imp = tp.importance.MagnitudeImportance(p=1, group_reduction='sum', normalizer=None)
    elif crit_name == 'BN':
        imp = tp.importance.BNScaleImportance(group_reduction='sum', normalizer=None)
    elif crit_name == 'FPGM':
        imp = tp.importance.FPGMImportance(group_reduction='sum', normalizer=None)
    elif crit_name == 'WHC':
        imp = WHCImportance(group_reduction='sum', normalizer=None)
    elif crit_name == 'Random':
        imp = tp.importance.RandomImportance()
    elif crit_name == 'Taylor':
        imp = tp.importance.TaylorImportance(group_reduction='sum', normalizer=None)
    elif crit_name == 'Jacobian':
        imp = GroupJacobianImportance_accumulate(group_reduction='sum', normalizer=None)
        
    # 2. Get a completely new model and Pruner
    model, pruner = get_model_and_pruner(imp, target_sparsity, pruning_steps)
    
    # 3. Start the iterative pruning loop
    for step in range(pruning_steps):
        current_sparsity_ratio = (step + 1) * (target_sparsity / pruning_steps)
        print(f"\n--- {crit_name} Pruning Step {step+1}/{pruning_steps} (Target Sparsity: ~{current_sparsity_ratio*100:.1f}%) ---")
        
        # Gradient accumulation (only needed for Taylor and Jacobian)
        if crit_name in ['Taylor', 'Jacobian']:
            model.eval()
            
            # Jacobian must reset score and grad externally every round to prevent overlap
            if crit_name == 'Jacobian':
                if hasattr(imp, 'zero_score'): imp.zero_score()
                if hasattr(imp, 'zero_grad'): imp.zero_grad()
                
            # Taylor relies on param.grad direct accumulation, so clear them uniformly at the beginning
            if crit_name == 'Taylor':
                model.zero_grad()
            
            for k, (imgs, lbls) in enumerate(train_loader):
                if k >= num_batches_for_grad:
                    break
                imgs = imgs.to(device)
                batch_texts = [f"a photo of a {classes[l]}" for l in lbls]
                text_toks = tokenizer(batch_texts).to(device)
                
                # Jacobian calculates JTJ internally for a single batch, so parameters must be zeroed before the batch
                if crit_name == 'Jacobian':
                    model.zero_grad()
                    
                # OpenCLIP Forward and manual logits reconstruction
                image_features, text_features, logit_scale = model(imgs, text_toks)
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logits_per_image.T
                
                labels = torch.arange(len(imgs)).to(device)
                loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
                loss.backward()
                
                if crit_name == 'Jacobian':
                    imp.accumulate_grad(model) # Accumulate Jacobian features
                    imp.accumulate_score(model) # Accumulate Jacobian features
                    torch.cuda.empty_cache()
                
        # Execute the structural pruning step
        pruner.step()
        
        # Test accuracy and parameter count
        pruned_acc = evaluate(model, test_loader)
        pruned_macs, pruned_params = count_parameters(model)
        
        results_acc[crit_name].append(pruned_acc)
        results_params[crit_name].append(pruned_params)
        results_macs[crit_name].append(pruned_macs)
        
        print(f"Result: {crit_name} Step {step+1} -> MACs Left {pruned_macs/1e9:.2f}G, Params Left: {pruned_params/1e6:.2f}M, Accuracy: {pruned_acc * 100:.2f}%")
        
        payload = {
            "criteria_names": criteria_names,
            "target_sparsity": target_sparsity,
            "pruning_steps": pruning_steps,
            "orig_acc": orig_acc,
            "orig_macs": orig_macs,
            "orig_params": orig_params,
            "results_acc": results_acc,
            "results_macs": results_macs,
            "results_params": results_params,
        }

        np.save(os.path.join(save_dir, "pruning_experiment_results.npy"), payload)


print("\nAll experiments completed!")