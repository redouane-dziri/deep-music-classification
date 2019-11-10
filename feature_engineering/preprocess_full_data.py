import os
from git_root import git_root
from google.cloud import storage
import librosa
import pandas as pd
import shutil
import sys
import json
from tqdm import tqdm

from preprocessing_pipeline import preprocess_data

#Load the utils module
sys.path.append(os.path.join(git_root(),'utils'))
from utils import load_config

#Load the config file
config = load_config()

### Read data from Google cloud storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arnaudstiegler/Desktop/Divers/W4111-e02930f7e70f.json"
storage_client = storage.Client("Music-Genre-Classification")
bucket = storage_client.get_bucket("deep-music-classification")


#Create a temporary folder for storing the current file
temp_dir_path = os.path.join(git_root(),'temp')

try:
    os.mkdir(temp_dir_path)
except:
    #Make sure the folder does not already exists
    shutil.rmtree(temp_dir_path)
    #then create it
    os.mkdir(temp_dir_path)

#Creating the dictionnary that will be given as input to the pipeline
data = {}
data['train'] = []
data['test'] = []

#Get the blobs for train/test
blobs_train = bucket.list_blobs(prefix=os.path.join("data","full_data","train"))
blobs_test = bucket.list_blobs(prefix=os.path.join("data","full_data","test"))

blobs_list = [blobs_train,blobs_test]

#We loop over train and test
for blobs in blobs_list:
    print("Reading a blob list")
    #for each blob (i.e file)
    for blob in tqdm(blobs):
        desc = blob.name.split('/')
        train_or_test = desc[2]
        label = desc[3]
        filename = desc[4]

        #Download the file from the blob
        with open(os.path.join(git_root(),'temp','temp_import.wav'),'wb') as infile:
            blob.download_to_file(infile)
        
        #Read the temporary file 
        file_numpy_representation, _ = librosa.load(os.path.join(git_root(),'temp','temp_import.wav'))

        #add it to our data
        data[train_or_test].append((filename,file_numpy_representation,label))
        


#Checking the results
# data should be on par with the input to generate_short_term_pieces_from_dict()

print("Preprocessing the data")
#We preprocess the data
x = preprocess_data(pre_loaded_data= data, serialize=True)

#We dump the data
print("Dumping the spectrogram data")

deg_list = config['feature_engineering']['GLCM']['spectrogram']["angles_in_deg"]
for i in range(len(deg_list)):
    #Dumping the result to a json file
    with open(os.path.join(git_root(),'data','preprocessed_data','data_spectrogram_angle_{}.json'.format(deg_list[i])),'w') as outfile:
        json.dump(x['spectrogram'][i], outfile)

print("Dumping the mel map data")

deg_list = config['feature_engineering']['GLCM']['mel_map']["angles_in_deg"]
for i in range(len(deg_list)):
    #Dumping the result to a json file
    with open(os.path.join(git_root(),'data','preprocessed_data','data_mel_map_angle_{}.json'.format(deg_list[i])),'w') as outfile:
        json.dump(x['mel_map'][i], outfile)

#Delete the temporary folder
shutil.rmtree(temp_dir_path)

