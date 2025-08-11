import abc
import warnings
import torch
import torch.nn as nn
from torch_pruning import importance 
from torch_pruning.pruner import function
import typing
from torch_pruning.dependency import Group
import torch_pruning as tp


import numpy as np
from scipy.spatial import distance
from copy import deepcopy


 

class GroupJacobianImportance_accumulate(importance.GroupMagnitudeImportance):
    """
    score = score_old + w.TJ.T J w, where score_old=w J_old.T J_old w;
    this way to accumulte J in several times so that the CUDA will no out-of-memory
    
    args:
    - normalize: ['parameters', and other options defined in 'torch_pruning.importance.GroupMagnitudeImportance']
        - 'parameters': importance=importance/filter_neumel
        - others: please refer to 'torch_pruning.importance.GroupMagnitudeImportance'
    -  others: please refer to 'torch_pruning.importance.GroupMagnitudeImportance'
     
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean',  # 'num_parameters'
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self._jacobian = {}
        self._counter = {} # useless actually
        self._importance = {}
        self._memory = {}

    def zero_grad(self):
        self._jacobian = {}
        self._counter = {}

    def zero_score(self):
        self._importance = {}
        
        
    def accumulate_grad(self, model, transposed=False):
        """
        Accumulate gradients to construct J.T@J
        Note here we don't consider the transposed convolution
        (note that the GroupHessianImportance don't as well, so be careful if you use it)
        
        
        model: literally model
        transposed: transposed layer or not; in transposed conv, the shape of weight is (in_channels, out_channels); 
                    while in normal conv, it is (out_channels, in_channels), and the code is based on this.
        """
        

        assert(transposed==False)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data.clone().unsqueeze(0) # (1,o,i,h,w) for conv; (1,o,i) for fc; (1,o) for the bias term

                if name not in self._jacobian:
                    self._jacobian[name] = grad # (n, o,i,h,w) for conv; (n, o,i) for fc; (n, o) for the bias term
                else:
                    self._jacobian[name] = torch.concat([self._jacobian[name],  grad], dim=0) # (n, o,i,h,w) for conv; (n, o,i) for fc; (n, o) for the bias term
                
                if name not in self._counter:
                    self._counter[name] = 1
                else:
                    self._counter[name] += 1

    def accumulate_score(self, model):
        # [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]
        for layer_name, layer in model.named_modules():
            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            if isinstance(layer, (nn.modules.conv._ConvNd, nn.Linear)):
                
                if layer.weight.grad is None:
                    warnings.warn(f"The gradient of {layer_name}.weight is None; please do loss.backward() first; otherwise no score for {layer_name} prune_xx_out_channels", UserWarning)
                else:
                    
                    '''Conv/Linear out_channels'''
                                       
                    if hasattr(layer, "transposed") and layer.transposed: # transposed convolution
                        assert(0) # Note here we don't consider the transposed convolution for ease of implementation
                    else:
                        w = layer.weight.data.clone().flatten(1) # (o, ihw); here idx is range(o) in Torch_pruning actually
                        j = self._jacobian[layer_name+'.weight']  # jacobian matrix of shape (n, o, ...)
                        j = j.reshape(j.shape[0], j.shape[1], -1) # (n, o, ihw) for conv; (n, o, i) for fc 
                        
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            w = torch.concat([w, layer.bias.data.clone().unsqueeze(1)], dim=1) # (o, ihw+1)
                            j_bias = self._jacobian[layer_name+'.bias'].unsqueeze(2) # (n, o, 1)
                            j = torch.concat([j, j_bias], dim=2) # (n,o, ihw+1)
                            
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(o); Resulting shape: (n, o)
                        jw = torch.einsum('nod,od->no', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(o); Resulting shape: (o,)
                        local_imp = torch.einsum('no,no->o', jw, jw)
                        
                        if layer_name+ '_out' not in self._importance:
                            self._importance[layer_name+ '_out'] = local_imp
                        else:
                            self._importance[layer_name+ '_out'] += local_imp
                    
                    
                    '''Conv/Linear in_channels'''             
                    if hasattr(layer, "transposed") and layer.transposed: # transposed convolution
                        assert(0) # Note here we don't consider the transposed convolution for ease of implementation
                    else:
                        w = layer.weight.data.clone().transpose(0, 1).flatten(1) # (i, ohw);
                        j = self._jacobian[layer_name+'.weight'].transpose(1,2)  # jacobian matrix of shape (n, i, o,...)
                        j = j.reshape(j.shape[0], j.shape[1], -1) # (n, i, ohw) for conv; (n, i, o) for fc 
                            
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(i); Resulting shape: (n, i)
                        jw = torch.einsum('nid,id->ni', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(i); Resulting shape: (i,) 
                        local_imp = torch.einsum('ni,ni->i', jw, jw)
                        
                        if layer_name+ '_in' not in self._importance:
                            self._importance[layer_name+ '_in'] = local_imp
                        else:
                            self._importance[layer_name+ '_in'] += local_imp
                        
            # BN
            elif  isinstance(layer, nn.modules.batchnorm._BatchNorm):
                if layer.affine:
                    if layer.weight.grad is None:
                        warnings.warn(f"The gradient of {layer_name}.weight is None; please do loss.backward() first; otherwise no score for {layer_name} prune_batchnorm_out_channels", UserWarning)
                    else:
                        w = layer.weight.data.clone() # (o); here idx is range(o) in Torch_pruning actually; so equal to 'w = layer.weight.data'
                        b = layer.bias.data.clone() # (o)
                        w = torch.stack([w, b], dim=1) # (o, 2)
                        
                        j_w = self._jacobian[layer_name+'.weight'] # (n, o)
                        j_b = self._jacobian[layer_name+'.bias']  # (n, o)
                        j = torch.stack([j_w, j_b], dim=2) # (n, o, 2)
                        
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(o); Resulting shape: (n, o)
                        jw = torch.einsum('nod,od->no', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(o); Resulting shape: (o,)
                        local_imp = torch.einsum('no,no->o', jw, jw)
                        
                        if layer_name+ '_out' not in self._importance:
                            self._importance[layer_name+ '_out'] = local_imp
                        else:
                            self._importance[layer_name+ '_out'] += local_imp
                                      
            # LN；
            # simply the copy the batchnorm here
            elif  isinstance(layer, nn.LayerNorm):
                if layer.elementwise_affine:
                    if layer.weight.grad is None:
                        warnings.warn(f"The gradient of {layer_name}.weight is None; please do loss.backward() first; otherwise no score for {layer_name} prune_layernorm_out_channels", UserWarning)
                    else:                        
                        w = layer.weight.data.clone() # (o); here idx is range(o) in Torch_pruning actually; so equal to 'w = layer.weight.data'
                        b = layer.bias.data.clone() # (o)
                        w = torch.stack([w, b], dim=1) # (o, 2)
                        
                        j_w = self._jacobian[layer_name+'.weight'] # (n, o)
                        j_b = self._jacobian[layer_name+'.bias']  # (n, o)
                        j = torch.stack([j_w, j_b], dim=2) # (n, o, 2)
                        
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(o); Resulting shape: (n, o)
                        jw = torch.einsum('nod,od->no', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(o); Resulting shape: (o,)
                        local_imp = torch.einsum('no,no->o', jw, jw)
                        
                        if layer_name+ '_out' not in self._importance:
                            self._importance[layer_name+ '_out'] = local_imp
                        else:
                            self._importance[layer_name+ '_out'] += local_imp         
            
                
        ###### Note: clear the accumulate_grad here ###
        self.zero_grad()
        
                
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        group_parameter_numel = 0 # the number of parameters of each group
 
        for i, (dep, idxs) in enumerate(group):
            idxs.sort() 
            layer = dep.target.module
            layer_name = dep.target.name[:dep.target.name.find(' ')]
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            
            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed: # transposed convolution
                        assert(0) # Note here we don't consider the transposed convolution for ease of implementation
                    else:
                        local_imp = self._importance[layer_name+ '_out'][idxs]
                    
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)      
                    group_parameter_numel += layer.weight.data.numel()/layer.weight.data.shape[0]
                
                    
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed: # transposed convolution
                        assert(0) # Note here we don't consider the transposed convolution for ease of implementation
                    else:
                        local_imp = self._importance[layer_name+ '_in'][idxs]
                        
                        # I don't know what this exactly for; just copy from GroupHessianImportance 
                        if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                            local_imp = local_imp.repeat(layer.groups)
                                                  
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)   
                    group_parameter_numel += layer.weight.data.numel()/layer.weight.data.shape[1]
                                            
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:

                        local_imp = self._importance[layer_name+ '_out'][idxs]
                        
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)    
                        group_parameter_numel += layer.weight.numel()*2 # w and bias
                        
            # LN；The correctness is not guranted since I am not familar with LayerNorm
            # simply the copy the batchnorm here
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if layer.weight.grad is not None:
                        
                        local_imp = self._importance[layer_name+ '_out'][idxs]
                        
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)    
                        group_parameter_numel += layer.weight.numel()*2 # w and bias

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        
        if self.normalizer=='parameters':
            group_imp = group_imp/group_parameter_numel
        else:
            group_imp = self._normalize(group_imp, self.normalizer)
            
        return group_imp



class GroupJacobianImportance(importance.GroupMagnitudeImportance):
    """
    score = w.T J.TJ w
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='parameters', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self._jacobian = {}
        self._counter = {}

    def zero_grad(self):
        self._jacobian = {}
        self._counter = {}
 
    def accumulate_grad(self, model, transposed=False):
        """
        Accumulate gradients to construct J.T@J
        Note here we don't consider the transposed convolution
        (note that the GroupHessianImportance don't as well, so be careful if you use it)
        
        
        model: literally model
        in_or_out: prune the input dim (channels) or output dim (filters)        
        transposed: transposed layer or not; in transposed conv, the shape of weight is (in_channels, out_channels); 
                    while in normal conv, it is (out_channels, in_channels), and the code is based on this.

        # bias: consider the bias term or not; IF YES, we shall concat it with the conv term in the Jacobian matrix when calculating score!
        """
        
        # another efficient way is calculate JTJ = old_JTJ + new_J.T@J
        # but it shall depend on the prune_fn; eg, if prune in_channels, we need to transpose the gradient
        # assert(transposed==False)
        # for name, param in model.named_parameters():
        #     if param.grad is not None and 'weight' in name:
        #         if name not in self._jt_j:
        #             self._jt_j[name] = 0 # jacobian.T @ jacobian
                    
        #         grad = param.grad.data.clone() # (o,i,h,w) for conv; (o,i) for fc
        #         if in_or_out=='in':
        #             grad = grad.transpose(0, 1) # (i,o,h,w) for conv; (i,o) for fc
        #         grad = grad.reshape(grad.shape[0],-1)
        #         self._jt_j[name] += grad[:, :, None] @ grad[:, None, :] # (o, ihw, ihw) or (i, ohw, ohw) for conv; (o,i,i) or (i,o,o) for fc

        #         if name not in self._counter:
        #             self._counter[name] = 1
        #         else:
        #             self._counter[name] += 1
        assert(transposed==False)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data.clone().unsqueeze(0) # (1,o,i,h,w) for conv; (1,o,i) for fc; (1,o) for the bias term

                if name not in self._jacobian:
                    self._jacobian[name] = grad # (n, o,i,h,w) for conv; (n, o,i) for fc; (n, o) for the bias term
                else:
                    self._jacobian[name] = torch.concat([self._jacobian[name],  grad], dim=0) # (n, o,i,h,w) for conv; (n, o,i) for fc; (n, o) for the bias term
                
                if name not in self._counter:
                    self._counter[name] = 1
                else:
                    self._counter[name] += 1

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        group_parameter_numel = 0 # the number of parameters of each group        

        for i, (dep, idxs) in enumerate(group):
            idxs.sort() 
            layer = dep.target.module
            layer_name = dep.target.name[:dep.target.name.find(' ')]
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            
            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed: # transposed convolution
                        assert(0) # Note here we don't consider the transposed convolution for ease of implementation
                    else:
                        w = layer.weight.data.clone()[idxs].flatten(1) # (o, ihw); here idx is range(o) in Torch_pruning actually
                        j = self._jacobian[layer_name+'.weight']  # jacobian matrix of shape (n, o, ...)
                        j = j.reshape(j.shape[0], j.shape[1], -1) # (n, o, ihw) for conv; (n, o, i) for fc 
                        
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            w = torch.concat([w, layer.bias.data.clone().unsqueeze(1)], dim=1) # (o, ihw+1)
                            j_bias = self._jacobian[layer_name+'.bias'].unsqueeze(2) # (n, o, 1)
                            j = torch.concat([j, j_bias], dim=2) # (n,o, ihw+1)
                            
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(o) 
                        # Resulting shape: (n, o)
                        jw = torch.einsum('nod,od->no', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(o) 
                        # Resulting shape: (o,)
                        local_imp = torch.einsum('no,no->o', jw, jw)
                        
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)   
                        group_parameter_numel += layer.weight.data.numel()/layer.weight.data.shape[0]           
                
                    
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed: # transposed convolution
                        assert(0) # Note here we don't consider the transposed convolution for ease of implementation
                    else:
                        w = layer.weight.data.clone().transpose(0, 1).flatten(1) # (i, ohw);
                        j = self._jacobian[layer_name+'.weight'].transpose(1,2)  # jacobian matrix of shape (n, i, o,...)
                        j = j.reshape(j.shape[0], j.shape[1], -1) # (n, i, ohw) for conv; (n, i, o) for fc 
                            
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(i) 
                        # Resulting shape: (n, i)
                        jw = torch.einsum('nid,id->ni', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(i) 
                        # Resulting shape: (i,)
                        local_imp = torch.einsum('ni,ni->i', jw, jw)
                        
                        # I don't know what this exactly for; just copy from GroupHessianImportance 
                        if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                            local_imp = local_imp.repeat(layer.groups)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)   
                        group_parameter_numel += layer.weight.data.numel()/layer.weight.data.shape[1]   
                                            
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data.clone()[idxs] # (o); here idx is range(o) in Torch_pruning actually; so equal to 'w = layer.weight.data'
                        b = layer.bias.data.clone()[idxs] # (o)
                        w = torch.stack([w, b], dim=1) # (o, 2)
                        
                        j_w = self._jacobian[layer_name+'.weight'] # (n, o)
                        j_b = self._jacobian[layer_name+'.bias']  # (n, o)
                        j = torch.stack([j_w, j_b], dim=2) # (n, o, 2)
                        
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(o) 
                        # Resulting shape: (n, o)
                        jw = torch.einsum('nod,od->no', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(o) 
                        # Resulting shape: (o,)
                        local_imp = torch.einsum('no,no->o', jw, jw)
                        
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs) 
                        group_parameter_numel += layer.weight.data.numel()*2   
                        
            # LN；The correctness is not guranted since I am not familar with LayerNorm
            # simply the copy the batchnorm here
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data.clone()[idxs] # (o); here idx is range(o) in Torch_pruning actually; so equal to 'w = layer.weight.data'
                        b = layer.bias.data.clone()[idxs] # (o)
                        w = torch.stack([w, b], dim=1) # (o, 2)
                        
                        j_w = self._jacobian[layer_name+'.weight'] # (n, o)
                        j_b = self._jacobian[layer_name+'.bias']  # (n, o)
                        j = torch.stack([j_w, j_b], dim=2) # (n, o, 2)
                        
                        # Step 1: Compute j[:, index, :] @ w[index, :].T for each index in range(o) 
                        # Resulting shape: (n, o)
                        jw = torch.einsum('nod,od->no', j, w)

                        # Step 2: Compute (j[:, index, :] @ w[index, :].T)^2 and sum over n for each index in range(o) 
                        # Resulting shape: (o,)
                        local_imp = torch.einsum('no,no->o', jw, jw)
                        
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)    
                        group_parameter_numel += layer.weight.data.numel()*2   
            

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        
        if self.normalizer=='parameters':
            group_imp = group_imp/group_parameter_numel
        else:
            group_imp = self._normalize(group_imp, self.normalizer)
            
        return group_imp





class WHCImportance(tp.importance.GroupMagnitudeImportance):

    def __init__(self, p=2, group_reduction="mean", normalizer='mean', bias=False):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)

                # L2 norm
                l2_norm = w.norm(p=2, dim=1)
                l2_norm_diag = torch.diag(l2_norm)

                # Cosine similarity matrix
                w_normed = torch.nn.functional.normalize(w, p=2, dim=1)
                similar_matrix = 1 - torch.abs(torch.matmul(w_normed, w_normed.T))  # Cosine similarity

                # Multiply with L2 norm diagonal matrix
                similar_matrix = l2_norm_diag @ similar_matrix @ l2_norm_diag
                similar_sum = similar_matrix.sum(dim=0)

                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.flatten(1)
                else:
                    w = layer.weight.data.transpose(0, 1).flatten(1)

                # L2 norm
                w = w.cpu()
                l2_norm = w.norm(p=2, dim=1)
                l2_norm_diag = torch.diag(l2_norm)

                # Cosine similarity matrix
                w_normed = torch.nn.functional.normalize(w, p=2, dim=1)
                similar_matrix = 1 - torch.abs(torch.matmul(w_normed, w_normed.T))  # Cosine similarity

                # Multiply with L2 norm diagonal matrix
                similar_matrix = l2_norm_diag @ similar_matrix @ l2_norm_diag
                local_imp = similar_matrix.sum(dim=0)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)

                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp



