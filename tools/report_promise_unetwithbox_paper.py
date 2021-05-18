import os
import numpy as np
import pandas as pd

def performance_summary(folder, dice_2d_all, T=None):
    epochs = list(dice_2d_all.keys())
    threshold = list(dice_2d_all[epochs[0]].keys())
    assert len(epochs)==50, len(epochs)
    mean_2d_array, std_2d_array = [], []
    for key in dice_2d_all.keys():
        mean_2d_array.append(np.mean(np.asarray(dice_2d_all[key]),axis=0))
        std_2d_array.append(np.std(np.asarray(dice_2d_all[key]),axis=0))
    mean_2d_array = np.vstack(mean_2d_array)
    std_2d_array = np.vstack(std_2d_array)
    if T is None:
        max_mean = np.max(mean_2d_array)
        ind = np.where(mean_2d_array==np.max(mean_2d_array))
        max_std  = std_2d_array[ind][0]
        epoch = epochs[ind[0][0]]
        th = threshold[ind[1][0]]
        # print('theshold: ', len(threshold))
        # print(len(epochs))
        print('{:s}: {:.3f}({:.3f}), th={:0.3f}, epoch={:s}'.format(folder, max_mean, max_std, th, epoch))
    else:
        loc = np.where(np.abs(np.asarray(threshold)-T)<1e-4)[0][0]
        mean_v = mean_2d_array[:,loc]
        std_v = std_2d_array[:,loc]
        max_mean = np.max(mean_v)
        ind = np.where(mean_v==np.max(mean_v))
        max_std  = std_v[ind[0][0]]
        epoch = epochs[ind[0][0]]
        print('{:s}: {:.3f}({:.3f}), th={:0.3f}, epoch={:s}'.format(folder, max_mean, max_std, T, epoch))
        
if __name__ == "__main__":
    dir_save_root = os.path.join('results','promise')
    folders = [## fs
               'residual_all_fs',
               ## mil baseline
               'residual_all_unary_pair', 
               ## ablation study - generalized mil
               'residual_parallel_focal_40_10_unary_pair',
               'residual_parallel_focal_40_20_unary_pair',
               'residual_parallel_focal_60_30_unary_pair',
               ## ablation study - smooth max approximation
               'residual_all_unaryapprox_expsumr=4_pair',
               'residual_all_unaryapprox_expsumr=6_pair',
               'residual_all_unaryapprox_expsumr=8_pair',
               'residual_all_unaryapprox_explogs=4_pair',
               'residual_all_unaryapprox_explogs=6_pair',
               'residual_all_unaryapprox_explogs=8_pair',
               ## main results
               'residual_parallel_approx_focal_40_20_expsumr=4_unary_pair',
               'residual_parallel_approx_focal_40_20_expsumr=6_unary_pair', 
               'residual_parallel_approx_focal_40_20_expsumr=8_unary_pair',
               'residual_parallel_approx_focal_40_20_explogs=4_unary_pair',
               'residual_parallel_approx_focal_40_20_explogs=6_unary_pair', 
               'residual_parallel_approx_focal_40_20_explogs=8_unary_pair',
               ]
  
    metrics = ['dice_3d']
    for metric in metrics:
        print('performance summary: {:s}'.format(metric).center(50,"#"))
        for folder in folders:
            output_dir = os.path.join(dir_save_root, folder)
            file_name = os.path.join(output_dir, metric+'.xlsx')
            results = pd.read_excel(file_name, sheet_name=None)
            performance_summary(folder, results)