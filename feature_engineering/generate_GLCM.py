from git_root import git_root
import os
import pandas as pd
import librosa
import numpy as np
import json
from skimage.feature import greycomatrix

#TODO: add argument to generate_dict_mapping so that it can work for the sample data or not

def generate_dict_mapping(sampling_rate):
	'''
	Function to load in memory the data from the sample_data folder

	returns: dict with 2 keys (train and test), the values being tuples of (file_name, numpy array, label)
	'''

	data_folder = os.path.join(git_root(),'data')
	df = pd.read_csv(os.path.join(data_folder,'metadata','train_test_split.csv'))

	train_df = df.loc[(df['split']=='train') & (df['sample'])]
	test_df = df.loc[(df['split']=='test') & (df['sample'])]
	
	train_records = []

	#Have to iterate over the rows to read files
	for index, row in train_df.iterrows():
		path = os.path.join(data_folder,'sample_data',row['split'],row['genre'],row['file_name'])
		file,sr = librosa.load(path, sr = sampling_rate)
		#Format of each tuple: (name, numpy array, label)
		train_records.append((row['file_name'],file,row['genre']))
	
	test_records = []

	for index, row in test_df.iterrows():
		path = os.path.join(data_folder,'sample_data',row['split'],row['genre'],row['file_name'])
		file,sr = librosa.load(path, sr = sampling_rate)

		#Format of each tuple: (name, numpy array, label)
		test_records.append((row['file_name'],file,row['genre']))


	return {'train': train_records, 'test': test_records}

def generate_short_term_piece(file_array,number_pieces=14, sampling_rate=22050, piece_length=4, overlap=2):
	'''
	Function to divide each music piece into short-term pieces (see paper)

	As a default, we have a 4sec piece with 2 seconds of overlap. The default sampling rate is the same as Librosa default.

	returns: numpy array of size (number_pieces, piece_length*sampling_rate)
	'''

	#The sampling rate is the number of frame per second
	frame_length = sampling_rate*piece_length #the number of frames for one short-term piece
	frame_overlap = overlap*sampling_rate #The number of frames of overlap
	frame_step = frame_length - frame_overlap

	sh = (file_array.size - frame_length + 1, frame_length)
	st = file_array.strides * 2
	view = np.lib.stride_tricks.as_strided(file_array, strides = st, shape = sh)[0::frame_overlap]
	
	return view.copy()

def check_windowing(w=4, o=2):
	'''
	Test function for the windowing process used in generate_short_term_piece()
	'''

	a = np.array([1,2,3,4,5,6,7,8])
	sh = (a.size - w + 1, w)
	st = a.strides * 2
	view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]

	return view.copy()


def generate_mel_maps(short_term_pieces, hop_length=2756, n_mels=40):

	mel_maps = np.array([librosa.feature.melspectrogram(piece,hop_length=2756,center=False, n_mels=40) for piece in short_term_pieces])

	return mel_maps


if __name__ == '__main__':

	with open(git_root("config", "config.json"), 'r') as config:
		config = json.load(config)

	params = config["feature_engineering"]

	data_dict = generate_dict_mapping(params["sampling_rate"])

	train_records = data_dict['train']

	#Testing for a single file

	file = train_records[0][1]
	print(file.shape) # (661794,)

	short_term_pieces = generate_short_term_piece(file,number_pieces=params["number_pieces"], 
															sampling_rate=params["sampling_rate"], 
															piece_length=params["piece_length_in_s"], 
															overlap=params["overlap_in_s"])
	print(short_term_pieces.shape) #(14, 88200)

	mel_maps = generate_mel_maps(short_term_pieces, hop_length=params["hop_length_in_s"]*params["sampling_rate"],
														n_mels=params["n_mels"])
	print(mel_maps.shape) #(14, 40, 32)

	#Next steps	
	#greycomatrix(image_scaled, [1], [0], levels=16).shape

	#For testing computation time

	for i in range(len(train_records)):

		file = train_records[i][1]
		short_term_pieces = generate_short_term_piece(file, number_pieces=params["number_pieces"], 
															sampling_rate=params["sampling_rate"], 
															piece_length=params["piece_length_in_s"], 
															overlap=params["overlap_in_s"])
		mel_maps = generate_mel_maps(short_term_pieces, hop_length=params["hop_length_in_s"]*params["sampling_rate"],
														n_mels=params["n_mels"])
