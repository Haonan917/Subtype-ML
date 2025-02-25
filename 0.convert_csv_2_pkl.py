'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-12-30 02:02:34
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-12-30 02:14:26
FilePath: /CGZSubtype-Predictor/0.convert_csv_2_pkl.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
from pathlib import Path
import time
import pickle
if __name__ == "__main__":
    data_dir = "data/subgroup_data/cg/"
    train_dir = "Training"
    test_dir = "Validation"
    
    data_dir_path = Path(data_dir)
    discovery_dir_path = data_dir_path / train_dir
    validation_dir_path = data_dir_path / test_dir
    
    omics_file_list = ["AllData_CG", "OmicsData_CG", "SubgroupData_CG"]
    from_suffix = ".csv"
    to_suffix = ".pkl"
    
    
    s = time.time()
    for omics_file in omics_file_list:
        all_train_file  = discovery_dir_path / (omics_file + from_suffix)
        all_validation_file  = validation_dir_path / (omics_file + from_suffix)

        # read csv
        all_train_df = pd.read_csv(all_train_file, sep="," ,index_col=0) # (feature, sample)
        all_validation_df = pd.read_csv(all_validation_file, sep=",",index_col=0) # (feature, sample)
        print(all_train_df.shape)
        print(all_validation_df.shape)

        # export pkl
        all_train_df.to_pickle(discovery_dir_path / (omics_file + to_suffix))
        all_validation_df.to_pickle(validation_dir_path / (omics_file + to_suffix))

    e = time.time()
    print(f"Time to run：{e-s} seconds")
    print("Finish converting")