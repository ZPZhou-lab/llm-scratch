#!/bin/bash

# path for the dataset
repo="HuggingFaceFW/fineweb-edu"
local_dir="fineweb-edu"

# file list for chinese
base_path="sample/10BT"
files=(
    "000_00000.parquet"
    "001_00000.parquet"
    # "002_00000.parquet"
    # "003_00000.parquet"
    # "004_00000.parquet"
    # "005_00000.parquet"
    # "006_00000.parquet"
    # "007_00000.parquet"
    # "008_00000.parquet"
    # "009_00000.parquet"
    # "010_00000.parquet"
    # "011_00000.parquet"
    # "012_00000.parquet"
    # "013_00000.parquet"
)

# set endpoint
export HF_ENDPOINT=https://hf-mirror.com
# download
for file in "${files[@]}"; do
    huggingface-cli download --repo-type dataset --resume-download $repo $base_path/$file --local-dir $local_dir
done