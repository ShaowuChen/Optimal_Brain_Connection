# [Optimal Brain Connection: Towards Efficient Structural Pruning](https://arxiv.org/abs/2508.05521)
[framework3.pdf](https://github.com/user-attachments/files/21719398/framework3.pdf)


<img width="413"  alt="framework" src="https://github.com/user-attachments/assets/7634c62f-a1ee-4564-899f-8a1ddd2fde64" />

~~I am currently focusing on my manuscript; the code will be organized and uploaded in about 10 days.~~

~~Update: The manuscript has been uploaded to arXiv. The code is planned to be uploaded before August 11th.~~

Partial code for Jacobian Criterion has been uploaded. Organizing complete code, may take some while (as a procrastinator on his vacation).


`Importance.py` implements our Jacobian Criterion and the compared data-free WHC based on the Torch_pruning framwork; more compared methods are contained in Torch_pruning.importance

`benchmark_criteria.py` provides a demo for comparing criteria.


To run code:

```
pip install torch-pruning --upgrade

wget https://github.com/VainF/Torch-Pruning/releases/download/v1.1.4/cifar100_vgg19.pth

CUDA_VISIBLE_DEVICES=0 python benchmark_criteria.py --pth_path ./cifar100_vgg19.pth --data_root ./ --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.9 --iterative_steps 18 
```
