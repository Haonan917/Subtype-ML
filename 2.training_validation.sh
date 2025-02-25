#!/bin/bash
###
 # @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @Date: 2024-12-30 02:41:27
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2024-12-30 09:02:10
 # @FilePath: /CGZSubtype-Predictor/2.training_validation.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

output_dir="./output_predictivity/"
feature_selection_dir='./predefined_features/Subgroup'
data_config_dir='./config/cgz_subgroup_config/cg'

seed_start=1
seed_end=10
seed_interval=1
endpoint_name="y.90"

for ((seed=$seed_start; seed<=$seed_end; seed+=$seed_interval)); do
    echo "$endpoint_name-$dataset-$seed"

    python training_code/run_train.py \
        --data_config_path "$data_config_dir/select_1/All.yaml" \
        --endpoint $endpoint_name \
        --output_dir $output_dir \
        --seed $seed \
        --return_with_mismatch \
        --dataset "all"

    python validation_code/run_validation.py \
        --data_config_path "$data_config_dir/select_1/All.yaml" \
        --endpoint $endpoint_name \
        --output_dir $output_dir \
        --seed $seed \
        --return_with_mismatch \
        --dataset "all"

    python training_code/run_train.py \
        --data_config_path "$data_config_dir/select_1/Omi.yaml" \
        --endpoint $endpoint_name \
        --output_dir $output_dir \
        --seed $seed \
        --return_with_mismatch \
        --dataset "omics"

    python validation_code/run_validation.py \
        --data_config_path "$data_config_dir/select_1/Omi.yaml" \
        --endpoint $endpoint_name \
        --output_dir $output_dir \
        --seed $seed \
        --return_with_mismatch \
        --dataset "omics"

    python training_code/run_train.py \
        --data_config_path "$data_config_dir/select_1/Sub.yaml" \
        --endpoint $endpoint_name \
        --output_dir $output_dir \
        --seed $seed \
        --return_with_mismatch \
        --dataset "subtype"

    python validation_code/run_validation.py \
        --data_config_path "$data_config_dir/select_1/Sub.yaml" \
        --endpoint $endpoint_name \
        --output_dir $output_dir \
        --seed $seed \
        --return_with_mismatch \
        --dataset "subtype"

done

