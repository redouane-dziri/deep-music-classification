import os
from git_root import git_root
from google.cloud import storage
import librosa
import pandas as pd
import shutil
import sys
import json
from tqdm import tqdm


#Load the utils module
sys.path.append(os.path.join(git_root(),'utils'))
from utils import load_config

#TODO: account for type and angle in fetch_data


def fetch_data(type='melmap', angle=0):
    #Load the config file
    config = load_config()

    ### Read data from Google cloud storage
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arnaud.stiegler/Desktop/Divers/adl-hw4-675afec62d41.json"
    storage_client = storage.Client("Music-Genre-Classification")
    bucket = storage_client.get_bucket("deep-music-classification")

    #Creating the dictionnary that will be given as input to the pipeline
    data = {}
    data['train'] = []
    data['test'] = []

    #Get the blobs for train/test
    #blobs_train = bucket.list_blobs(prefix=os.path.join("data","preprocessed_data",type))
    #df = pd.read_json(os.path.join("gs://deep-music-classification","data","preprocessed_data","mel_map","data_mel_map_angle_90.jsonß"))

    blob = bucket.get_blob(os.path.join("data","preprocessed_data","mel_map","data_mel_map_angle_90.json"))
    content = blob.download_as_string()
    my_df = pd.read_json(content)  

    return my_df

if __name__=="__main__":
    df = fetch_data()
    print(df['train'].shape)
