import os

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/red/Documents/ADL/Project/My First Project-9a16875b5624.json"

storage_client = storage.Client("Music-Genre-Classification")

bucket = storage_client.get_bucket("deep-music-classification")

blobs = bucket.list_blobs(prefix=os.path.join("data","full_data","test"))

for blob in blobs:
    print(blob)