import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')
df=pd.read_parquet(r"C:\Users\gonth\Downloads\0000 (1).parquet")
#%%capture
song_vectorizer = CountVectorizer()
song_vectorizer.fit(df['track_genre'])
df= df.sort_values(by=['popularity'], ascending=False).head(10000)
def get_similarities(song_name, data):

  # Getting vector for the input song.
  text_array1 = song_vectorizer.transform(data[data['track_name']==song_name]['track_genre']).toarray()
  num_array1 = data[data['track_name']==song_name].select_dtypes(include=np.number).to_numpy()

  # We will store similarity for each row of the dataset.
  sim = []
  for idx, row in data.iterrows():
	  name = row['track_name']
	
	  # Getting vector for current song.
	  text_array2 = song_vectorizer.transform(data[data['track_name']==name]['track_genre']).toarray()
	  num_array2 = data[data['track_name']==name].select_dtypes(include=np.number).to_numpy()

	 # Calculating similarities for text as well as numeric features
	  text_sim = cosine_similarity(text_array1, text_array2)[0][0]
	  num_sim = cosine_similarity(num_array1, num_array2)[0][0]
	  sim.append(text_sim + num_sim)
	
  return sim
st.title(":rainbow[Spotify Recommondation model]")
def recommend_songs(song_name, data=df):
  # Base case
  if df[df['track_name'] == song_name].shape[0] == 0:
    st.write('This song is either not so popular or you\
    have entered invalid_name.\n Some songs you may like:\n')
     
    for song in data.sample(n=5)['track_name'].values:
      print(song)
    return
   
  data['similarity_factor'] = get_similarities(song_name, data)
 
  data.sort_values(by=['similarity_factor', 'popularity'],
                   ascending = [False, False],
                   inplace=True)
   
  # First song will be the input song itself as the similarity will be highest.
  display(data[['track_name', 'artists']][2:7])
s=st.text_input("enter song")
if s in df:
    recommend_songs(s)
    

