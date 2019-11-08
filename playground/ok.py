import os

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arnaudstiegler/Desktop/Divers/W4111-e02930f7e70f.json"

storage_client = storage.Client("Music-Genre-Classification")

bucket = storage_client.get_bucket("deep-music-classification")

blobs = bucket.list_blobs(prefix=os.path.join("data","full_data","test"))

for blob in blobs:
    print(blob)
    desc = blob.name.split('/')
    train_or_test = desc[2]
    label = desc[3]
    filename = desc[4]
   

#Snippet to understand how to download/upload files from google storage
'''
for blob in blobs:
    with open(os.path.join(git_root(),'playground','test.wav'),'wb') as infile:
        blob.download_to_file(infile)
    blob2 = bucket.blob('remote/path/storage.txt')
    blob2.upload_from_filename(filename='/local/path.txt')
    break

x = librosa.load(os.path.join(git_root(),'playground','test.wav'))
print(x)
'''