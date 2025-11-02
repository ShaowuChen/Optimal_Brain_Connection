import os

import pandas as pd

def save_dir_name_pruned(args):
    global_way = 'global' if args.global_pruning==True else 'local'
    reg = 0 if args.reg==False else args.reg

    args.output_dir = (
    f'./results/{args.output_dir}/{args.dataset}_{args.mode}/'
    f'{args.model}/{args.method}_Eq{args.equivalent}_SL{args.sparsity_learning}_Gl{global_way}_Step{args.iterative_steps}/GR{args.group_reduction}_No{args.normalizer}_NB{args.N_batchs}/'
    f'fr{args.speed_up}_dr{args.delete_rate}_er{args.expand_rate}_id{args.initi_div}_'
    f'l{args.coeff_label}_T{args.T}_c{args.coeff_ce}_Reg{reg}/'
    f'{args.save_suffix}')
        
    os.makedirs(args.output_dir, exist_ok=True)

    result_name = (f'{args.model}_fr{args.speed_up}_dr{args.delete_rate}_Eq{args.equivalent}_{args.method}_SL{args.sparsity_learning}_Gl{global_way}_GR{args.group_reduction}_No{args.normalizer}_NB{args.N_batchs}_'
                    f'er{args.expand_rate}_id{args.initi_div}_'
                    f'l{args.coeff_label}_T{args.T}_c{args.coeff_ce}_Reg{reg}')
                    # _s'
                    # f'{args.save_suffix}')
                        
    # result_name = (
    #     f'{args.model}_{args.importance_criterion}_{args.method}_{global_way}_GR{args.group_reduction}_No{args.normalizer}_NB{args.N_batchs}_'
    #     f'fr{args.speed_up}_dr{args.delete_rate}_er{args.expand_rate}_id{args.initi_div}_'
    #     f'l{args.coeff_label}_T{args.T}_f{args.coeff_feature}_c{args.coeff_ce}_s{args.save_suffix}')
                                        
    exp_file = f'./{args.output_dir}/{result_name}.log'

    # excel_path = f'{args.output_dir}/../../{args.model}_results_{args.data_set}_{args.save_suffix}.xlsx'
    excel_path = f'./{args.output_dir}/{args.model}_{args.method}_Eq{args.equivalent}_SL{args.sparsity_learning}_Gl{global_way}_GR{args.group_reduction}_No{args.normalizer}_{args.save_suffix}.xlsx'
    
    args.result_name = result_name
    args.exp_file = exp_file
    args.excel_path = excel_path
    
    return args, result_name, exp_file, excel_path  

def check_and_add_row(file_path, row_index,  key_word, save_suffix, value):
    # lock_path = file_path + ".lock"  # 创建锁文件
    # with FileLock(lock_path):
    # Check if file exists
    if os.path.exists(file_path):
        # Read the excel file
        df = pd.read_excel(file_path, index_col=0, engine='openpyxl')
    else:
        # Create a DataFrame with specified columns if the file does not exist
        df = pd.DataFrame(columns=[
            'Speedup_0', 'Speedup_1', 'Speedup_2','Speedup_AVE','Speedup_STD',
            'Params_rate_0', 'Params_rate_1', 'Params_rate_2', 'Params_rate_AVE','Params_rate_STD',
            'SL_top1_0', 'SL_top1_1', 'SL_top1_2', 'SL_top1_AVE','SL_top1_STD',
            'SL_best_top1_0', 'SL_best_top1_1', 'SL_best_top1_2', 'SL_best_top1_AVE','SL_best_top1_STD',
            'raw1_0', 'raw1_1', 'raw1_2', 'raw1_AVE', 'raw1_STD', 
            'top1_0', 'top1_1', 'top1_2', 'top1_AVE', 'top1_STD',
            'EP_top1_0', 'EP_top1_1', 'EP_top1_2', 'EP_top1_AVE', 'EP_top1_STD',
            'best_top1_0', 'best_top1_1', 'best_top1_2', 'best_top1_AVE', 'best_top1_STD',
            'EP_best_top1_0', 'EP_best_top1_1', 'EP_best_top1_2', 'EP_best_top1_AVE', 'EP_best_top1_STD',
            'SL_top5_0', 'SL_top5_1', 'SL_top5_2', 'SL_top5_AVE','SL_top5_STD',
            'raw5_0', 'raw5_1', 'raw5_2', 'raw5_AVE', 'raw5_STD', 
            'top5_0', 'top5_1', 'top5_2', 'top5_AVE', 'top5_STD',
            'EP_top5_0', 'EP_top5_1', 'EP_top5_2', 'EP_top5_AVE', 'EP_top5_STD',
            'evaluate_time_0', 'evaluate_time_1', 'evaluate_time_2', 'evaluate_time_AVE','evaluate_time_STD'
        ])
        

    # Update the specified cell
    df.loc[row_index, key_word + '_' + str(save_suffix)] = value

    # Calculate averages for row_index '2' only if the first three cells are not NaN
    # if save_suffix == '2':
    #     for keyword in ['Speedup', 'Params_rate', 'SL_top1', 'raw1', 'SL_top1', 'raw5', 'top1', 'best_top1', 'top5', 'EP_top1', 'EP_best_top1','EP_top5', 'evaluate_time']:
    #         values = df.loc[row_index, [keyword + '_0', keyword + '_1', keyword + '_2']].dropna().astype(float)
    #         if len(values) == 3:
    #             df.loc[row_index, keyword + '_AVE'] = values.mean().round(2)
    #             df.loc[row_index, keyword + '_STD'] = values.std().round(2)
                
    # Save the DataFrame back to the excel file
    df.to_excel(file_path)
