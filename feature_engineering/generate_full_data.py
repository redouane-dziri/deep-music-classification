import os
from git_root import git_root
from google.cloud import storage
import librosa
import pandas as pd
import shutil


### Read data from Google cloud storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arnaudstiegler/Desktop/Divers/W4111-e02930f7e70f.json"

storage_client = storage.Client("Music-Genre-Classification")

bucket = storage_client.get_bucket("deep-music-classification")


#Create a temporary folder for storing the files
temp_dir_path = os.path.join(git_root(),'temp')

try:
    os.mkdir(temp_dir_path)
except:
    #Make sure the folder does not already exists
    shutil.rmtree(temp_dir_path)
    #then create it
    os.mkdir(temp_dir_path)

data = {}
data['train'] = []
data['test'] = []

blobs_train = bucket.list_blobs(prefix=os.path.join("data","full_data","train"))
blobs_test = bucket.list_blobs(prefix=os.path.join("data","full_data","test"))

blobs_list = [blobs_train,blobs_test]

#We download the file
for blobs in blobs_list:
    for blob in blobs:
        desc = blob.name.split('/')
        train_or_test = desc[2]
        label = desc[3]
        filename = desc[4]
        print(train_or_test)

        with open(os.path.join(git_root(),'temp','temp_import.wav'),'wb') as infile:
            blob.download_to_file(infile)
        
        #Read the temporary file
        np_array = librosa.load(os.path.join(git_root(),'temp','temp_import.wav'))
        data[train_or_test].append((filename,np_array,label))


#Checking the results
# data should be on par with the input to generate_short_term_pieces_from_dict()
print(len(data['train']))
print(len(data['test']))

#Delete the temporary folder
shutil.rmtree(temp_dir_path)

