# IMPORTANT: train_test_split should have been run before

import os
import shutil

import json

import numpy as np
import pandas as pd

from git_root import git_root

with open(git_root("config", "config.json"), 'r') as config:
    config = json.load(config)

data_root = git_root("data")

train_test_split_path = os.path.join(
    data_root, "metadata", "train_test_split.csv"
)

train_test_split = pd.read_csv(train_test_split_path)

random_state = 11

def sample_18_2(block):
    sample_size = 2
    if block["split"].unique()[0] == "train":
        sample_size = 18
    return block.sample(sample_size, random_state=random_state)

samples = train_test_split.groupby(["genre", "split"]).\
    apply(sample_18_2).\
    reset_index(drop=True)

sample_data_path = git_root("data", "sample_data")
full_data_path = git_root("data", "full_data")

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

for split in ["train", "test"]:
    make_dir(os.path.join(sample_data_path, split))
    for genre in config["genres"]:
        make_dir(os.path.join(sample_data_path, split, genre))

def copy_file(row):
    shutil.copy(
        os.path.join(
            full_data_path, row["split"], row["genre"], row["file_name"]
        ), 
        os.path.join(
            sample_data_path, row["split"], row["genre"], row["file_name"]
        )
    )

# WARNING: don't un-comment unless you want to change the whole samples 
# irreversibly

_ = samples.apply(copy_file, axis=1)

# add sample information to train_test_split.csv 

train_test_split = train_test_split[["file_name", "genre", "split"]].merge(
    samples[["file_name"]], how="left", on="file_name", indicator="sample"
)

train_test_split["sample"] = train_test_split["sample"].map(
    {"left_only": False, "both": True}
)

# WARNING: don't un-comment unless you want to change the whole samples 
# irreversibly

train_test_split.to_csv(train_test_split_path, index=False)
