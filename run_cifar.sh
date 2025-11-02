data_root="/your/data/root"  # please set your data root path here

# define restore path
# ! Please download the pretrained models first and place them in the path
# mkdir -p ./results/base_model/
# wget -P ./results/base_model/  https://github.com/ShaowuChen/Optimal_Brain_Connection/releases/download/v0/cifar100_vgg19.pth
# wget -P ./results/base_model/  https://github.com/ShaowuChen/Optimal_Brain_Connection/releases/download/v0/cifar10_resnet56.pth

declare -A restore_paths
restore_paths["vgg19_cifar100"]="./results/base_model/cifar100_vgg19.pth"
restore_paths["resnet56_cifar10"]="./results/base_model/cifar10_resnet56.pth"

# I also upload the models after sparse training, you can download them and place them in
# wget -P ./results/base_model/  https://github.com/ShaowuChen/Optimal_Brain_Connection/releases/download/v0/reg_cifar100_vgg19_Jacobian_0.0005.pth
# wget -P ./results/base_model/  https://github.com/ShaowuChen/Optimal_Brain_Connection/releases/download/v0/reg_cifar10_resnet56_Jacobian_0.0005.pth
resotre_sparse_paths["vgg19_cifar100"]="./results/base_model/reg_cifar100_vgg19_Jacobian_0.0005.pth"
resotre_sparse_paths["resnet56_cifar10"]="./results/base_model/reg_cifar10_resnet56_Jacobian_0.0005.pth"


# I have 8 Titan Xp GPUs; control concurrency; # Single GPU for each model; # Each GPU run 2 tasks at a time
# Please adjust the max_parallel_jobs according to your GPU number and memory
max_parallel_jobs=16  
         

#######################################
# Experiments on CIFAR-100 with VGG19
#######################################
job_count=0 
models=("vgg19")
datasets=("cifar100")
speed_ups=("1.5" "3.0" "6" "9" "12") # For Table IV and IX
normalizers=("None")
methods=("Jacobian" "l2") # Our method VS. DepGraph
save_suffixes=(0 1 2) # eachh experiment run 3 times
group_reductions=("max" "sum" "first")

lr_divs=(5)  # adjust lr for C and D
wd_divs=(1)  # adjust wd for C and D

for method in "${methods[@]}"; do 
  for normalizer in "${normalizers[@]}"; do
    for model_index in "${!models[@]}"; do
      
      model="${models[$model_index]}"
      dataset="${datasets[$model_index]}"

      # use original pretrained model and do sparse training first
      restore_path="${restore_paths[${model}_${dataset}]}"
      ## Or directly use the sparse trained model to save time
      # sl_restore_path="${resotre_sparse_paths[${model}_${dataset}]}"

      for speed_up in "${speed_ups[@]}"; do
        for lr_div in "${lr_divs[@]}"; do
          for wd_div in "${wd_divs[@]}"; do
            for group_reduction in "${group_reductions[@]}"; do
              for save_suffix in "${save_suffixes[@]}"; do
                
                # assign GPU 0~7
                gpu_id=$((job_count % 8))  
                
                echo CUDA_VISIBLE_DEVICES=$gpu_id python main_cifar.py \
                  --dataroot "$data_root" \
                  --model "$model" \
                  --dataset "$dataset" \
                  --speed-up "$speed_up" \
                  --save-suffix "$save_suffix" \
                  --restore "$restore_path" \
                  --finetune \
                  --method "$method" \
                  --equivalent \
                  --N-batchs 50 \
                  --mode prune \
                  --lr-div "$lr_div" \
                  --wd-div "$wd_div" \
                  --select "CB" \
                  --normalizer "$normalizer" \
                  --group-reduction "$group_reduction" \
                  --normalizer_for_sl  "max"\
                  --output-dir "vgg" \
                  --sparsity-learning 
                  #--sl-restore "$sl_restore_path" 

                job_count=$((job_count + 1))

                # every max_parallel_jobs, wait until all tasks done
                if (( job_count % max_parallel_jobs == 0 )); then
                  wait  
                fi
              done
            done
          done
        done
      done
    done
  done
done

wait # wait until all tasks done
echo $job_count



#######################################
# Experiments on CIFAR-100 with ResNet-56
#######################################
job_count=0 
models=("resnet56")
datasets=("cifar10")
speed_ups=("2.11" "2.55") # For Table IV
normalizers=("None")
methods=("Jacobian")
save_suffixes=(0 1 2) # run 3 times each experiment

lr_divs=(0.5) # adjust lr for C and D, 2x
wd_divs=(0) # weight decay=0 for C and D

for method in "${methods[@]}"; do 
  for normalizer in "${normalizers[@]}"; do
    for model_index in "${!models[@]}"; do
      
      model="${models[$model_index]}"
      dataset="${datasets[$model_index]}"

      # use original pretrained model and do sparse training first
      restore_path="${restore_paths[${model}_${dataset}]}"
      ## Or directly use the sparse trained model to save time
      # sl_restore_path="${resotre_sparse_paths[${model}_${dataset}]}"
      
      for speed_up in "${speed_ups[@]}"; do
        for lr_div in "${lr_divs[@]}"; do
          for wd_div in "${wd_divs[@]}"; do
            for save_suffix in "${save_suffixes[@]}"; do
                
                # assigang GPU 0~7
                gpu_id=$((job_count % 8)) 
                
                echo CUDA_VISIBLE_DEVICES=$gpu_id python main_cifar.py \
                --dataroot "$data_root" \
                --model "$model" \
                --dataset "$dataset" \
                --speed-up "$speed_up" \
                --save-suffix "$save_suffix" \
                --restore "$restore_path" \
                --finetune \
                --method "$method" \
                --global-pruning \
                --equivalent \
                --N-batchs 50 \
                --mode prune \
                --lr-div "$lr_div" \
                --wd-div "$wd_div" \
                --select "CB" \
                --normalizer "$normalizer" \
                --group-reduction "sum" \
                --normalizer_for_sl  "max"\
                --output-dir "resnet56"\
                --sparsity-learning 
                #--sl-restore "$sl_restore_path" 

                job_count=$((job_count + 1))

                # every max_parallel_jobs, wait until all tasks done
                if (( job_count % max_parallel_jobs == 0 )); then
                wait 
                fi
              
            done
          done
        done
      done
    done
  done
done


wait
echo $job_count