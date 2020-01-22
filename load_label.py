import numpy as np
import pickle as cP
import random
import itertools
import os
import threading
from scipy import sparse
import pandas as pd


def sort_coo(m):
	tuples = zip(m.row, m.col, m.data)
	return sorted(tuples, key=lambda x: (x[2]), reverse=True)

def load_label(args):


	#song_user_str = './data/song_user_matrix_20000_10000.npz'
	song_user_str = './data/song_user_matrix_1000_100.npz'
	song_user_csr = sparse.load_npz(song_user_str)
	song_user_coo = sparse.coo_matrix(song_user_csr)


	# find negative examples
	# (item, user)
	# first make user's item list (already composed)
	# except that randomly generate negative samples
	df = pd.DataFrame({'item':song_user_coo.row,'user':song_user_coo.col,'data':song_user_coo.data})


	# load co list
	co_list,Sid_to_Tid,D7id_to_path,Tid_to_D7id,songs =	get_co_list(args.feature_path,args.num_song)
	
	# load split
	train_idx, valid_idx, test_idx = split(args.num_song, co_list)
	print(df.shape)

	# filtering matrix
	df_train = df[df['item'].isin(train_idx)] #1523904 x 3
	print(df_train.shape)
	
	user_to_item_train = df_train.groupby('user')['item'].apply(list)
	item_to_user_train = df_train.groupby('item')['user'].apply(list)

	song_user_coo_train = sparse.coo_matrix(sparse.csr_matrix((df_train.values[:,2].astype(int),(df_train.values[:,0],df_train.values[:,1]))))
	print(song_user_coo_train.shape)

	sorted_coo_train = sort_coo(song_user_coo_train)

	df_valid = df[df['item'].isin(valid_idx)]
	print(df_valid.shape)
	
	item_to_user_valid  = df_valid.groupby('item')['user'].apply(list)
	user_to_item_valid = df_valid.groupby('user')['item'].apply(list)

	song_user_coo_valid = sparse.coo_matrix(sparse.csr_matrix(((df_valid.values[:,2].astype(int),(df_valid.values[:,0],df_valid.values[:,1])))))
	print(song_user_coo_valid.shape)

	sorted_coo_valid = sort_coo(song_user_coo_valid)


	all_items = co_list
	print('label loaded')


	return sorted_coo_train, sorted_coo_valid, songs, user_to_item_train, user_to_item_valid, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, item_to_user_train,item_to_user_valid, train_idx, valid_idx, test_idx


def get_co_list(feature_path,num_song):
	# data load
	Sid_to_Tid = cP.load(open('./data/echonest_id_to_MSD_id.pkl','rb'))
	D7id_to_path = cP.load(open('./data/7D_id_to_path.pkl','rb'))
	Tid_to_D7id = cP.load(open('./data/MSD_id_to_7D_id.pkl','rb'))
	song_str = './data/subset_songs_20000_10000.npy'
	songs = np.load(song_str)[0:100]

	# read files in audio subdirectories
	audio_list = []
	for path, subdirs, files in os.walk(feature_path):
		for name in files:
			if not name.startswith('.'):
				tmp = os.path.join(path,name).replace('\\', '/')
				tmp = tmp.split('/')
				audio_list.append(tmp[9]+'/'+tmp[10]+'/'+tmp[11].replace('.npy','.mp3'))
	print(len(audio_list))
    

	path_to_D7id = dict(list(zip(list(D7id_to_path.values()), list(D7id_to_path.keys())))) # ipath - 7digital 8/7/8731992.clip.mp3  8731992
	D7id_to_Tid = dict(list(zip(list(Tid_to_D7id.values()), list(Tid_to_D7id.keys())))) # 7digital echonest 5504670 TRAAAAK128F9318786 
	Tid_to_Sid = dict(list(zip(list(Sid_to_Tid.values()), list(Sid_to_Tid.keys())))) # echonest MSD TRBGKMB128F4257851 to SOAAABI12A8C13615F 

	songs_list=[]           
	for i in range(0,len(songs)):
		songs_list.append(songs[i].decode('UTF-8'))	

	Sid_audio = []
	for iter in range(len(audio_list)):
		try:
			a = Tid_to_Sid[D7id_to_Tid[path_to_D7id[audio_list[iter]]]]
			for item in songs_list:
				if a in item:
					Sid_audio.append(a) #MSD [SOAAABI12A8C13615F,...]
					break
				else:
					continue
		except Exception:
			continue
	#Sid_audio = []
	#for iter in range(len(audio_list)):
	#	try:
	#		Sid_audio.append(Tid_to_Sid[D7id_to_Tid[path_to_D7id[audio_list[iter]]]]) #MSD [SOAAABI12A8C13615F,...]
	#	except Exception:
	#		continue
        
	# compare with songs
    

        
	idx_to_songs = dict(list(zip(np.arange(num_song), songs_list))) #songs msd  [1,SOAAABI12A8C13615F] in the user song matrix
	songs_to_idx = dict(list(zip(list(idx_to_songs.values()), list(idx_to_songs.keys()))))  #  [SOAAABI12A8C13615F,1] change dic
    
    #co_songs = list(set(Sid_audio)&set(songs)) # MSD [SOAAABI12A8C13615F,SOAAABI12A8C13615F ]
	co_list = []
	for iter in range(len(Sid_audio)):
		co_list.append(songs_to_idx[Sid_audio[iter]]) #  (songs_to_idx[co_songs[1]  get [1,2,3,4..]

	return co_list,Sid_to_Tid,D7id_to_path,Tid_to_D7id,songs 


def split(num_song,co_list):
	# train / valid / test split (7/1/2)
	split_t = 7
	split_v = 8
	total = np.arange(num_song)
	train_idx = []
	for iter in range(split_t):
		train_idx.append(np.where(total%10==iter)[0])
	train_idx = list(itertools.chain(*train_idx))
	valid_idx = np.where(total%10==split_v)[0]
	test_idx = []
	for iter in range(split_v,10):
		test_idx.append(np.where(total%10==iter)[0])
	test_idx = list(itertools.chain(*test_idx))
	print(len(train_idx),len(valid_idx),len(test_idx))

	train_idx = list(set(train_idx) & set(co_list))
	valid_idx = list(set(valid_idx) & set(co_list))
	test_idx = list(set(test_idx) & set(co_list))
	print(len(train_idx),len(valid_idx),len(test_idx))
	return train_idx, valid_idx, test_idx

