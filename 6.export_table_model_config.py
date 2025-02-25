'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-11-06 03:31:41
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-01-15 08:08:15
FilePath: /CGZMain-Predictor/3.sumup_predictivity.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './biomarker_code/')))
import pandas as pd
import argparse
from pathlib import Path
import re
import os
import yaml
import biomarker_code
from sklearn.metrics import  auc, roc_auc_score
import pickle
    
ENPOINT_PATTERN = "y.[\d]+"
BINARY_ENDPOINTS = ['y.90', 'y.158']
FEATURES_DICTS = {
    "all": pd.read_csv("data/subgroup_data/cg/Training/AllData_CG.csv", sep=",", index_col=0).index.to_list(),
    "omics": pd.read_csv("data/subgroup_data/cg/Training/OmicsData_CG.csv", sep=",", index_col=0).index.to_list(),
    "subtype": pd.read_csv("data/subgroup_data/cg/Training/SubgroupData_CG.csv", sep=",", index_col=0).index.to_list()
}

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_feature_names(pipe, numeric_features, categorical_features):
    numeric_feature_names = numeric_features
    if len(categorical_features) != 0:
        encoder = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        categorical_feature_names = encoder.get_feature_names_out(categorical_features)
    else:
        categorical_feature_names = []
    
    # 合并数值特征和分类特征的列名
    all_feature_names = list(numeric_feature_names) + list(categorical_feature_names)
    
    return all_feature_names
def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    endpoints = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    print(FEATURES_DICTS)
    # calculate mean auc curve for ensemble_algorithms
    PATTERN = r"output_refits_(?P<omics_type>.*?).pkl"
    for endpoint in endpoints:
        if not endpoint.startswith('y.'):
            continue
        endpoint_dir = os.path.join(output_dir, endpoint)
        valid_results = {}
        test_results = {}
        seeds = [d for d in os.listdir(endpoint_dir) if os.path.isdir(os.path.join(endpoint_dir, d))]
        
        all_config_df = None
        all_kbest_df = {}
        for seed in seeds:
            seed_dir = os.path.join(endpoint_dir, seed)
            files =  [f for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
            model_files = [f for f in files if f.startswith('output_refits')]
            omics_list = list(set([re.match(PATTERN, f).group("omics_type") for f in model_files]))

            for omics in omics_list:
                model_path = Path(os.path.join(seed_dir, 'output_refits_{}.pkl'.format(omics)))
                refit_models = pickle.load(open(model_path, 'rb'))
                config_dict = {}
                config_dict["Seed"] = int(seed.split('_')[-1])
                config_dict["Modality"] = omics

                if omics not in all_kbest_df:
                    all_kbest_df[omics] = None
                    
                for algo in refit_models.keys():
                    if algo == 'avg':
                        continue

                    pipeline = refit_models[algo][0]
                    model = pipeline.named_steps[algo]
                    kbest = pipeline.named_steps["kbest"]

                    ori_feats = FEATURES_DICTS[omics]
                    cat_feats = ['subgroup'] if 'subgroup' in ori_feats else []
                    con_feats = [col for col in ori_feats if col not in cat_feats]
                    all_feats = get_feature_names(pipeline, con_feats, cat_feats)

                    selected_features = kbest.get_support()
                    kbest_df = pd.DataFrame({'{}_{}'.format(algo, config_dict["Seed"]): selected_features}, index=all_feats)
                    all_kbest_df[omics] = pd.concat([all_kbest_df[omics], kbest_df], axis=1)
    
                    params = model.get_params()
                    if algo == "svc":
                        config_dict["svm.C"]= params["C"]
                        config_dict["svm.kbest"]= kbest.k


                    elif algo == "lr":
                        config_dict["lr.l1_ratio"]= params["l1_ratio"]
                        config_dict["lr.C"] = params["C"]
                        config_dict["lr.kbest"]= kbest.k
                    elif algo == "rf":
                        config_dict["rf.kbest"]= kbest.k
                        config_dict["rf.max_features"] = params["max_features"]
                        config_dict["rf.n_estimators"]= params["n_estimators"]
                        config_dict["rf.max_depth"] = params["max_depth"]
                        config_dict["rf.min_samples_split"] = params["min_samples_split"]


                config_df = pd.DataFrame(config_dict, index=[0])
                all_config_df = pd.concat([all_config_df, config_df], ignore_index=True)

        sorted_df = all_config_df.sort_values(by=['Modality', 'Seed'], ascending=[True, True])
        sorted_df.to_csv(os.path.join(output_dir, 'table_config.csv'), index=False)

        for omics in omics_list:
            kbest_df = all_kbest_df[omics]
            sorted_columns = sorted(kbest_df.columns, key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
            sorted_kbest_df = kbest_df[sorted_columns]
            sorted_kbest_df.to_csv(os.path.join(output_dir, 'table_{}_kbest.csv'.format(omics)), index=True)

                
if __name__ == '__main__':
    dir_name = './output_predictivity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=dir_name, help="parent output directory")
    parser.add_argument('--output_dir', type=str, default=dir_name, help="parent output directory")
    args = parser.parse_args()
    main(args)
