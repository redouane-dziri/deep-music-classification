import os
from git_root import git_root
from google.cloud import storage
import pandas as pd
import sys


#Load the utils module
sys.path.append(os.path.join(git_root(),'utils'))
from utils import load_config


def fetch_data(map_type='mel_map', train=True, angle='0'):
    '''
    Function to fetch data from google storage

    BEWARE: the map_type should be in ['mel_map', 'spectrogram','mfcc']


    '''
    #Load the config file
    config = load_config()

    ### Read data from Google cloud storage
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arnaud.stiegler/Desktop/Divers/adl-hw4-675afec62d41.json"
    storage_client = storage.Client("Music-Genre-Classification")
    bucket = storage_client.get_bucket("deep-music-classification")

    if train:
        set_type = 'train'
    else:
        set_type = 'test'

    if(map_type=='mfcc'):
        blob_name = os.path.join("data","preprocessed_data",map_type,"data_{}_{}.json".format(map_type,set_type))
    else:
        blob_name = os.path.join("data","preprocessed_data",map_type,"data_{}_angle_{}_{}.json".format(map_type,angle,set_type))

    print("Fetching: {}".format(blob_name))

    blob = bucket.get_blob(blob_name)
    content = blob.download_as_string()
    df = pd.read_json(content)  

    if(map_type=='mfcc'):
        df.columns = ['filename','maps','genre']
    else:
        df.columns = ['filename','maps','split_id','genre']

    return df

if __name__=="__main__":
    df = fetch_data('spectrogram', '0', False)
    print(df.shape)
