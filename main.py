import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

# load in the data
moviesData = pd.read_csv('data/movies.csv', sep=',', names=['movieId', 'title', 'genres'])
ratingsData = pd.read_csv('data/ratings.csv', sep=',', names=['userId', 'movieId', 'rating', 'timestamp'])

# Merge movies and ratings on movieId
df = pd.merge(moviesData, ratingsData, on='movieId')
# remove duplicate column names
df = df.drop(0)

df = df.replace("?", None)

# convert rating data to float type
df["rating"] = pd.to_numeric(df["rating"])

# view head of CSV
print(df.head())

'''
average rating for each movie and the number of ratings
We are going to use these ratings to calculate the correlation between the movies later.
Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together.
Movies that have a high correlation coefficient are the movies that are most similar to each other.
'''
ratings = pd.DataFrame(df.groupby('title')['rating'].mean(), dtype=float)
print(ratings.head())

'''
Want to see the relationship between the average rating of a movie and the number of ratings the movie got.
It is very possible that a 5 star movie was rated by just one person.
It is therefore statistically incorrect to classify that movie has a 5 star movie.
So we need to set a threshold for the minimum number of ratings as we build the recommender system.
'''
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
print(ratings.head())

# visualize the distribution of the ratings
'''
We can see that most of the movies are rated between 2 and 4.5
'''
plt.title('Ratings Histogram')
plt.hist(ratings.rating, bins=50, density=True, color='orange')
plt.show()

'''
From the below histogram it is clear that most movies have few ratings.
Movies with most ratings are those that are most famous.
'''
plt.title('Number of Ratings Histogram')
plt.hist(ratings.number_of_ratings, bins=50, density=True, color='orange')
plt.show()

'''
Their is a positive relationship between the average rating of a movie and the number of ratings.
The graph indicates that the more the ratings a movie gets the higher the average rating it gets.
This is important to note especially when choosing the threshold for the number of ratings per movie.
'''
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
plt.show()

'''
In order to create our item based recommender system,
we need to convert our dataset into a matrix with the movie titles as the columns,
the userId as the index and the ratings as the values.
By doing this we shall get a dataframe with the columns as the movie titles and the rows as the user ids.
Each column represents all the ratings of a movie by all users.
The rating appear as NAN where a user didn't rate a certain movie.
We shall use this matrix to compute the correlation between the ratings of a single movie and the rest of the movies in the matrix.
'''
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
print(movie_matrix.head())

'''
Most rated movies and choose two of them to work.
Top 10 most rated movies
'''
print('--------- Top 10 rated movies ----------')
print(ratings.sort_values('number_of_ratings', ascending=False).head(10))

'''
Let’s assume that a user has watched Air Force One (1997) and Contact (1997).
We would like like to recommend movies to this user based on this watching history.
The goal is to look for movies that are similar to Contact (1997) and Air Force One (1997 which we shall recommend to this user.
We can achieve this by computing the correlation between these two movies’ ratings and the ratings of the rest of the movies in the dataset.
The first step is to create a dataframe with the ratings of these movies from our movie_matrix.
'''
forrest_gump_user_rating = movie_matrix['Forrest Gump (1994)']
pulp_fiction_user_rating = movie_matrix['Pulp Fiction (1994)']

print(forrest_gump_user_rating.head())
print(pulp_fiction_user_rating.head())

'''
Get the correlation between each movie's rating and the ratings of Forrest Hump & Pulp Fiction movies.
'''
similar_to_forrest_gump=movie_matrix.corrwith(forrest_gump_user_rating).sort_values(ascending=False)
similar_to_pulp_fiction=movie_matrix.corrwith(pulp_fiction_user_rating).sort_values(ascending=False)

'''
We can see that the correlation between Forrest Gump (1994) movie and The DUFF (2015) is 0.923.
This indicates a very strong similarity between these two movies.
'''
print('---------- Similar to Forrest Gump ----------')
print(similar_to_forrest_gump.head(700))

'''
We can see that the correlation between Pulp Fiction (1994) movie and Pet Sematary II (1992) is 0.866.
This indicates a very strong similarity between these two movies.
'''
print('---------- Similar to Pulp Fiction ----------')
print(similar_to_pulp_fiction.head(500))

'''
Drop all those null values and transform correlation results into dataframes to make the results look more appealing.
'''
corr_forest_gump = pd.DataFrame(similar_to_forrest_gump, columns=['correlation'])
corr_forest_gump.dropna(inplace=True)
print(corr_forest_gump.head(700))

corr_pulp_fiction = pd.DataFrame(similar_to_pulp_fiction, columns=['correlation'])
corr_pulp_fiction.dropna(inplace=True)
print(corr_pulp_fiction.head(500))

'''
Set a threshold for the number of ratings so we can get accurate recommendations.
To do so we need to join the two dataframes with the number_of_ratings column in the ratings dataframe
'''
corr_forest_gump = corr_forest_gump.join(ratings['number_of_ratings'])
corr_pulp_fiction = corr_pulp_fiction.join(ratings['number_of_ratings'])
print(corr_forest_gump)
print(corr_pulp_fiction)

'''
Obtain the movies that are most similar to Forrest Gump (1994) by limiting them to movies that have at least 100 reviews
We then sort them by the correlation column and view the first 10

We notice that Forrest Gump (1994) has a perfect correlation with itself, which is not surprising.
The next most similar movie is Good Will Hunting (1997) with a correlation of 0.484.
By changing the threshold for the number of reviews we get different results from the previous way of doing it.
Limiting the number of rating gives us better results
and we can confidently recommend the above movies to someone who has watched Forrest Gump (1994)
'''
print(corr_forest_gump[corr_forest_gump['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10))

'''
Once again, we obtain the movies that are most similar to Pulp Fiction (1994) by limiting them to movies that have at least 100 reviews
We then sort them by the correlation column and view the first 10

We notice that Pulp Fiction (1994) has a perfect correlation with itself, which is not surprising.
The next most similar movie is Fight Club (1999) with a correlation of 0.543.
By changing the threshold for the number of reviews we get different results from the previous way of doing it.
Limiting the number of rating gives us better results
and we can confidently recommend the above movies to someone who has watched Pulp Fiction (1994)
'''
print(corr_pulp_fiction[corr_pulp_fiction['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10))

