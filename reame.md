# Demo For AAAI Manuscript `Optimal Brain Connection: Towards Efficient Structural Pruning'

---

`Importance.py` implements our Jacobian Criterion and the compared data-free WHC based on the Torch_pruning framwork; more compared methods are contained in Torch_pruning.importance

`benchmark_criteria.py` provides a demo for comparing criteria.


To run code:

```
pip install torch-pruning --upgrade

wget https://github.com/VainF/Torch-Pruning/releases/download/v1.1.4/cifar100_vgg19.pth

CUDA_VISIBLE_DEVICES=0 python benchmark_criteria.py --pth_path ./cifar100_vgg19.pth --data_root ./ --repeats 5 --N_batchs 50 --global_pruning  --pruning_ratio 0.9 --iterative_steps 18 
```
