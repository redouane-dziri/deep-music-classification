import os
import shutil

import json

import numpy as np
import pandas as pd

from git_root import git_root

with open(git_root("config", "config.json"), 'r') as config:
    config = json.load(config)

output = git_root("data", "metadata", "train_test_split.csv")

# writes csv with three columns in `data/metadata`:
# - file_name: e.g. blues.00098.wav
# - genre: e.g. blues
# - split: train or test

output_dict = {
    "file_name": [],
    "genre": [],
    "split": [],
}

for genre in config["genres"]:

    all_indices = np.arange(0, 100)
    test_indices = np.random.choice(all_indices, 20, replace=False)
    train_indices = np.delete(all_indices, test_indices)

    test_files = [
        f"{genre}.000{str(index).zfill(2)}.wav" for index in test_indices
    ]
    train_files = [
        f"{genre}.000{str(index).zfill(2)}.wav" for index in train_indices
    ]

    output_dict["file_name"].extend(train_files)
    output_dict["genre"].extend([genre] * config["train_percent"])
    output_dict["split"].extend(["train"] * config["train_percent"])
    output_dict["file_name"].extend(test_files)
    output_dict["genre"].extend([genre] * (100 - config["train_percent"]))
    output_dict["split"].extend(["test"] * (100 - config["train_percent"]))

output_df = pd.DataFrame(output_dict)

# WARNING: don't un-comment unless you want to change the whole train/test split 
# irreversibly

# output_df.to_csv(output, index=False)

# move the files now

data_path = git_root("data", "full_data")

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

for split in ["train", "test"]:
    make_dir(os.path.join(data_path, split))
    for genre in config["genres"]:
        make_dir(os.path.join(data_path, split, genre))

def copy_file(row):
    shutil.copy(
        os.path.join(data_path, row["genre"], row["file_name"]), 
        os.path.join(data_path, row["split"], row["genre"], row["file_name"])
    )

# WARNING: don't un-comment unless you want to change the whole train/test split 
# irreversibly

# _ = output_df.apply(copy_file, axis=1)

# clean up

# WARNING: don't un-comment unless you want to change the whole train/test split 
# irreversibly

#for genre in config["genres"]:
#    shutil.rmtree(os.path.join(data_path, genre))
