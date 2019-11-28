import os
import sys

import numpy as np
import pandas as pd

from google.cloud import storage
from git_root import git_root

#Load the utils module
sys.path.append(git_root("utils"))
from utils import load_config, load_credentials, load_params

import tensorflow as tf



def fetch_data_cloud(map_type, angle=None, train=True):
	"""This function to fetches data from json files in google cloud storage and 
	returns a pandas DataFrame

	Arguments:
		map_type {string} -- one of ('spectrogram', 'mel_map', 'mfcc'), the type
			of map the GLCM was made of or the MFCC map `map`
		angle {string} -- the angle of the GLCM to fetch, in string form (e.g. 
			'0')
		train {boolean} -- whether to fetch the training or testing data

	Returns:
		df {pd DataFrame} -- if `map_type = 'mfcc'`, columns are 
			['filename', 'map', 'genre'] else 
			['filename', 'map', 'split_id', 'genre']
			where 'filename' is the name of the track, 'map' is the numeric
			numpy array representation, 'genre' is the track genre and 
			'split_id' is the id of the short-term piece the map was built on
	"""

	### Read data from Google cloud storage
	credentials = load_credentials()
	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials['PATH']
	storage_client = storage.Client("Music-Genre-Classification")
	bucket = storage_client.get_bucket("deep-music-classification")

	if train:
		set_type = "train"
	else:
		set_type = "test"

	if(map_type == "mfcc"):
		blob_name = os.path.join(
			"data", 
			"preprocessed_data",
			map_type,
			"data_{}_{}.json".format(map_type, set_type)
		)
	else:
		blob_name = os.path.join(
			"data",
			"preprocessed_data",
			map_type,
			"data_{}_angle_{}_{}.json".format(map_type, angle, set_type)
		)

	print("Fetching: {}".format(blob_name))

	blob = bucket.get_blob(blob_name)
	content = blob.download_as_string()
	df = pd.read_json(content)  

	if(map_type == "mfcc"):
		df.columns = ["filename", "map", "genre"]
	else:
		df.columns = ["filename", "map", "split_id", "genre"]

	return df


def fetch_data_local(map_type, angle=None, train=True):
	"""
	This function to fetches data from json files locally and returns a pandas 
	DataFrame

	Arguments:
		map_type {string} -- one of ('spectrogram', 'mel_map', 'mfcc'), the type
			of map the GLCM was made of or the MFCC map `map`
		angle {string} -- the angle of the GLCM to fetch, in string form (e.g. 
			'0')
		train {boolean} -- whether to fetch the training or testing data

	Returns:
		df {pd DataFrame} -- if `map_type = 'mfcc'`, columns are 
			['filename', 'map', 'genre'] else 
			['filename', 'map', 'split_id', 'genre']
			where 'filename' is the name of the track, 'map' is the numeric
			numpy array representation, 'genre' is the track genre and 
			'split_id' is the id of the short-term piece the map was built on
	"""

	if train:
		set_type = "train"
	else:
		set_type = "test"

	if(map_type == "mfcc"):
		file_name = git_root(
			"data",
			"preprocessed_data_full",
			map_type,
			"data_{}_{}.json".format(map_type, set_type)
		)
	else:
		file_name = git_root(
			"data",
			"preprocessed_data_full",
			map_type,
			"data_{}_angle_{}_{}.json".format(map_type, angle, set_type)
		)

	print("Fetching: {}".format(os.path.basename(file_name)))

	df = pd.read_json(file_name)  

	if(map_type == "mfcc"):
		df.columns = ["filename", "maps", "genre"]
	else:
		df.columns = ["filename", "maps", "split_id", "genre"]

	return df


def to_numpy_arrays(df, mfcc=False):
	"""This function takes in a dataframe with columns 
	['filename', 'maps', 'genre'] like one output by `fetch_data_local` or
	`fetch_data_cloud` and returns two np arrays `samples` and `labels`
	containing, respectively, the numpy maps and their associated labels
	
	Arguments:
		df {pd DataFrame} -- with columns ['filename', 'map', 'genre'] where 
			'filename' is the name of the track, 'map' is the numeric numpy 
			array representation, 'genre' is the track genre 
		mfcc {boolean} -- default is False, whether the maps are MFCC
	
	Returns:
		samples, labels {(list, list)} -- list of numpy arrays containing the
			maps and their associated labels
	"""
	config = load_config()
	params = load_params()

	label_names = config["genres"]

	if not mfcc:
		input_dim_1 = params["quantization"]["n_levels"] - 1
		input_dim_2 = input_dim_1
		input_dim_3 = 1
	else:
		input_dim_1 = params["MFCC"]["n_submaps"]
		input_dim_2 = params["MFCC"]["n_windows"] // input_dim_1
		input_dim_3 = params["MFCC"]["n_mfcc"]
		
	label_to_idx = dict((name, index) for index, name in enumerate(label_names))

	samples = []
	labels = []
	for _, row in df.iterrows():
		to_add = np.array(row['maps']).reshape(
			input_dim_1, input_dim_2, input_dim_3
		)
		to_add = np.swapaxes(to_add, 0, -1) if mfcc else to_add
		samples.append(to_add)
		labels.append(label_to_idx[row['genre']])
	
	return samples, labels


def prepare_tf_dataset(samples, labels):
	"""
	
	Arguments:
		samples {[type]} -- [description]
		labels {[type]} -- [description]
	
	Returns:
		[type] -- [description]
	"""
	BATCH_SIZE = 32

	dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
	dataset = dataset.shuffle(128).batch(BATCH_SIZE)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return dataset

def fetch_format_stacked(train):
	'''
	Succinct Function that fetches and stacks all angles for mel_map to create the stacked numpy array

	Args: None

	Returns: np.array of shape (1400,15,15,4) 
	'''
	params = load_params()

	angles = params["GLCM"]["mel_map"]["angles_in_deg"]
	n_angles = len(angles)
	levels_dim = params['quantization']['n_levels'] -1

	return np.squeeze([np.array(
	to_numpy_arrays(
		fetch_data_local(
			map_type="mel_map", train=train, angle=angle))[0]) for angle in angles]).reshape(-1,levels_dim,levels_dim,n_angles)



if __name__ == "__main__":
	print("Fetching from Google Storage")
	df = fetch_data_cloud("spectrogram", angle="0", train=False)
	print(df.shape)
	print("\n")
	print("Fetching from local files")
	df = fetch_data_local("spectrogram", angle="0", train=False)
	print(df.shape)
