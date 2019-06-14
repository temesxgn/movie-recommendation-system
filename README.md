# Movie Recommendation System


```
This is a very simple way of building a recommender system and is no where near close to industry standards.

This system can be improved by building a Memory-Based Collaborative Filtering based system.
In this case we’d divide the data into a training set and a test set.
We’d then use techniques such as cosine similarity to compute the similarity between the movies.
An alternative is to build a Model-based Collaborative Filtering system.
This is based on matrix factorization.
Matrix factorization is good at dealing with scalability and sparsity than the former.
You can then evaluate your model using techniques such as Root Mean Squared Error(RMSE).

There are other techniques for building recommender systems.
Deep learning is one of the ways of doing so especially when you have massive datasets.
Some of the algorithms that are being used to build advanced recommender systems include Autoencoders and Restricted Boltzmann machines

Followed: https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d
```
Common types of recommendation systems are:
* **_Content based_**:
    * Use meta data such as genre, producer, actor, musician to recommend items say movies or music.
        Such a recommendation would be for instance recommending Infinity War that featured Vin Disiel
        because someone watched and liked The Fate of the Furious. 
        Similarly you can get music recommendations from certain artists because you liked their music. 
        Content based systems are based on the idea that if you liked a certain item you are most likely to like something that is similar to it.
* **_Collaborative filtering_**: 
    * The behavior of a group of users is used to make recommendations to other users. 
        Recommendation is based on the preference of other users. 
        A simple example would be recommending a movie to a user based on the fact that their friend liked the movie.
    * Two types of collaborative models **_Memory-based_** methods and **_Model-based_**
    * **_Memory-based_**:
        * The advantage of memory-based techniques is that they are simple to implement and the resulting recommendations are often easy to explain. They are divided into two type: User-based & Item-based
        * User-based collaborative filtering: In this model products are recommended to a user based on the fact that the products have been liked by users similar to the user. For example if Derrick and Dennis like the same movies and a new movie comes out that Derick likes,then we can recommend that movie to Dennis because Derrick and Dennis seem to like the same movies.
        * Item-based collaborative filtering: These systems identify similar items based on users’ previous ratings. For example if users A,B and C gave a 5 star rating to books X and Y then when a user D buys book Y they also get a recommendation to purchase book X because the system identifies book X and Y as similar based on the ratings of users A,B and C
    * **_Model-based_**:
        * Based on matrix factorization and are better at dealing with sparsity. They are developed using data mining, machine learning algorithms to predict users’ rating of unrated items. In this approach techniques such as dimensionality reduction are used to improve the accuracy. Examples of such model-based methods include decision trees, rule-based models, Bayesian methods and latent factor models.