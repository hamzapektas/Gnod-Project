#!/usr/bin/env python
# coding: utf-8

# In[17]:


import spotipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from spotipy.oauth2 import SpotifyClientCredentials


# In[2]:


secrets_file = open("spotipy.txt","r")


# In[3]:


string = secrets_file.read()


# In[4]:


s = string.split('\n')


# In[5]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=s[0],
                                                           client_secret=s[1]))


# In[10]:


df_final = pd.read_csv('playlist.csv')


# In[11]:


topsongs = pd.read_csv('topsongs.csv')


# In[36]:


cluster_data = df_final.drop(['artist','name','uri','Unnamed: 0', 'cluster'], axis=1)


# In[37]:


scaler= StandardScaler().fit(cluster_data)
X_scaled = scaler.transform(cluster_data)


# In[46]:


# df_final


# In[40]:


kmeans = KMeans(n_clusters=5, random_state=1234)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)
pd.Series(clusters).value_counts().sort_index()


# In[19]:


def features_api(track, artist):
    track_id = sp.search(q='artist:' + artist + ' track:' + track, type='track')
    uri = track_id["tracks"]["items"][0]['id']
    features_api = sp.audio_features(uri)
    return features_api


# In[50]:


def recommend():
    new_song = input("Enter a song: ").lower()
    new_artist = input("Enter an artist: ").lower()
    try:
        
        if new_song in np.array(topsongs['song']):
            return random.choice(np.array(topsongs['song']))
        else:
            feature = features_api(new_song, new_artist)
            column = list(feature[0].keys())
            values = [list(feature[0].values())]
            df_new_song = pd.DataFrame(data = feature, columns = column)
            df_new_song = df_new_song.drop(['type','id','uri','track_href','analysis_url','duration_ms','time_signature'],axis=1)
            std_new_song = scaler.transform(df_new_song)
            new_cluster = kmeans.predict(std_new_song)
            df_cluster = df_final[df_final['cluster'] == list(new_cluster)[0]]
        

            print('Your recommendation:',random.choice(list(df_cluster['name'])))
    except:
         print('Ups! This song is not exist! Please try a new one', recommend()) 


# In[51]:


recommend()


# In[ ]:





# In[ ]:




