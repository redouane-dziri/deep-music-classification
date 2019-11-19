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
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arnaud.stiegler/Desktop/Divers/adl-hw4-675afec62d41.json"
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

output_dir = git_root("data", "pipeline_output")

train_test_list = ['train', 'test']

print("Dumping the MFCC")

    mfcc_dir_path = os.path.join(git_root(),'data','preprocessed_data','mfcc')

    try:
        os.mkdir(mfcc_dir_path)
    except:
        pass

    for elem in train_test_list:
            filename = 'data_mfcc_{}.json'.format(elem)
            print(filename)

            #Dumping the result to a json file
            with open(os.path.join(spectrogram_dir_path,filename),'w') as outfile:
                json.dump(x['mfcc'][elem], outfile)

            #Loading the data to google storage
            blob_out = bucket.blob(os.path.join("data","preprocessed_data","mfcc",filename))
            blob_out.upload_from_filename(filename=os.path.join(mfcc_dir_path,filename))

#We dump the data
print("Dumping the spectrogram data")

spectrogram_dir_path = os.path.join(git_root(),'data','preprocessed_data','spectrogram')

try:
    os.mkdir(spectrogram_dir_path)
except:
    pass

deg_list = config['feature_engineering']['GLCM']['spectrogram']["angles_in_deg"]
for i in range(len(deg_list)):

    for elem in train_test_list:
        filename = 'data_spectrogram_angle_{}_{}.json'.format(deg_list[i], elem)
        print(filename)

        #Dumping the result to a json file
        with open(os.path.join(spectrogram_dir_path,filename),'w') as outfile:
            json.dump(x['spectrogram'][i], outfile)

        #Loading the data to google storage
        blob_out = bucket.blob(os.path.join("data","preprocessed_data","spectrogram",filename))
        blob_out.upload_from_filename(filename=os.path.join(spectrogram_dir_path,filename))


print("Dumping the mel map data")

mel_map_dir_path = os.path.join(git_root(),'data','preprocessed_data','mel_map')

try:
    os.mkdir(mel_map_dir_path)
except:
    pass

deg_list = config['feature_engineering']['GLCM']['mel_map']["angles_in_deg"]
for i in range(len(deg_list)):

    for elem in train_test_list:
        filename = 'data_mel_map_angle_{}_{}.json'.format(deg_list[i], elem)
        print(filename)

        #Dumping the result to a json file
        with open(os.path.join(mel_map_dir_path,filename),'w') as outfile:
            json.dump(x['mel_map'][i][elem], outfile)
        #Loading the data to google storage
        blob_out = bucket.blob(os.path.join("data","preprocessed_data","mel_map",filename))
        blob_out.upload_from_filename(filename=os.path.join(mel_map_dir_path,filename))


#Delete the temporary folder
shutil.rmtree(temp_dir_path)

