import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# load in the data
moviesData = pd.read_csv('data/movies.csv', sep=',', names=['movieId', 'title', 'genres'])
ratingsData = pd.read_csv('data/ratings.csv', sep=',', names=['userId', 'movieId', 'rating', 'timestamp'])

# Merge movies and ratings on movieId
df = pd.merge(moviesData, ratingsData, on='movieId')

# remove duplicate column names
df = df.drop(0)

# view head of CSV
print(df.head())

'''
average rating for each movie and the number of ratings
We are going to use these ratings to calculate the correlation between the movies later.
Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together.
Movies that have a high correlation coefficient are the movies that are most similar to each other.
'''

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())
