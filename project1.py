# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:04:01 2020

@author: yukta
"""
#importing libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
#define a sample text
text=["london paris paris","london london paris"]
#cv is object of countvectorizer
cv=CountVectorizer()
#creating variable to save output of vectorization
count_matrix=cv.fit_transform(text)
#to array will convert matrix to array
print((count_matrix).toarray())
#variable 
similarity_scores=cosine_similarity(count_matrix)
print(similarity_scores)
#read dataset
df=pd.read_csv("movie_dataset.csv")
#data preprocessing
df=df.iloc[:,0:24]

#select some  features
features=['keywords','genres','cast','director']
# fill "NA" with blank spaces
for feature in features:
    df[feature]=df[feature].fillna('')
#create a cloumn in dataframe to combine features
def combine_features(row):
    return row['keywords']+""+row['genres']+""+row['cast']+""+row['director']
#create new column in dataset
df["combine_features"]=df.apply(combine_features,axis=1)
print(df["combine_features"].head())
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["combine_features"])
#similarity score
cosine_sim=cosine_similarity(count_matrix)
def get_title_from_index(index):
    return  df[df.index==index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]
#define a sample test
print((count_matrix).toarray())
movie_user_likes='Spider-Man 3'
#get index of movie
movie_index=get_index_from_title(movie_user_likes)
print(movie_index)
#list of tupple of similar movies
similar_movies=list(enumerate(cosine_sim[int(movie_index)]))
#sort of tupple
sorted_similar_movies=sorted(similar_movies,keys=lambda x:x[1],reverse=True)
#print title
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if i>5:
        break

