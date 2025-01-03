#!/bin/bash

# path for the dataset
repo="graelo/wikipedia"
local_dir="wikipedia"

# file list for chinese
zh_base_path="data/20230901/zh"
zh_files=(
    # "train-0001-of-0011.parquet"
    # "train-0002-of-0011.parquet"
    # "train-0003-of-0011.parquet"
    # "train-0004-of-0011.parquet"
    # "train-0005-of-0011.parquet"
    "train-0006-of-0011.parquet"
    "train-0007-of-0011.parquet"
    "train-0008-of-0011.parquet"
    # "train-0009-of-0011.parquet"
    # "train-0010-of-0011.parquet"
)

# set endpoint
export HF_ENDPOINT=https://hf-mirror.com
# download
for file in "${zh_files[@]}"; do
    huggingface-cli download --repo-type dataset --resume-download $repo $zh_base_path/$file --local-dir $local_dir
done

# # file list for english
# en_base_path="data/20230901/en"
# en_files=(
#     "train-0001-of-0084.parquet"
#     "train-0002-of-0084.parquet"
#     "train-0003-of-0084.parquet"
#     "train-0004-of-0084.parquet"
#     "train-0005-of-0084.parquet"
#     "train-0006-of-0084.parquet"
#     "train-0007-of-0084.parquet"
#     "train-0008-of-0084.parquet"
#     "train-0009-of-0084.parquet"
#     "train-0010-of-0084.parquet"
#     "train-0011-of-0084.parquet"
#     "train-0012-of-0084.parquet"
#     "train-0013-of-0084.parquet"
#     "train-0014-of-0084.parquet"
#     "train-0015-of-0084.parquet"
#     "train-0016-of-0084.parquet"
#     "train-0017-of-0084.parquet"
#     "train-0018-of-0084.parquet"
#     "train-0019-of-0084.parquet"
#     "train-0020-of-0084.parquet"
# )

# # download
# for file in "${en_files[@]}"; do
#     huggingface-cli download --repo-type dataset --resume-download $repo $en_base_path/$file --local-dir $local_dir
# done