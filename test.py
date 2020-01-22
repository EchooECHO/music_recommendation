# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:35:06 2019

@author: ECHO
"""

#test on delet user-item matrix

song_user_str = './data/song_user_matrix_20000_10000.npz'
song_user_csr = sparse.load_npz(song_user_str)
song_user_coo = sparse.coo_matrix(song_user_csr)


# find negative examples
# (item, user)
# first make user's item list (already composed)
# except that randomly generate negative samples
df_new = pd.DataFrame({'item':song_user_coo.row,'user':song_user_coo.col,'data':song_user_coo.data})
keep_song =list(range(0,100))

arr = np.arange(20000)
np.random.shuffle(arr)
keep_user =list(arr)
keep_user=keep_user[0:1000]


df_new = df_new[df_new['item'].isin(keep_song)]
df_new=df_new[df_new['user'].isin(keep_user)]

a = sparse.coo_matrix(sparse.csr_matrix(((df_new.values[:,2].astype(int),(df_new.values[:,0],df_new.values[:,1])))))
sparse.save_npz('song_user_matrix_1000_100.npz', a) 
df_new.groupby(df_new['user']).size() # user 19869


df_new.values[:,0]
