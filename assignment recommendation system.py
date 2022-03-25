# Q1)

import pandas as pd
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Recommendation Engine\game.csv")
data.shape
data.columns

# Exploratory data analysis

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
% matplotlib inline

# Let's create a ratings dataframe with average rating and number of ratings

data.groupby('game')['rating'].mean().sort_values(ascending= False).head(10)

data.groupby('game')['rating'].count().sort_values(ascending= False).head(10)

ratings = pd.DataFrame(data.groupby('game')['rating'].mean())
ratings.head()

# Now set the number of ratings column:
    
ratings['no_of_ratings'] = pd.DataFrame(data.groupby('game')['rating'].count())
ratings.head()

plt.figure(figsize= (5,4))
ratings['rating'].hist(bins=6)

plt.figure(figsize=(5,4))
ratings['no_of_ratings'].hist(bins=6)


sns.jointplot(x='rating',y='no_of_ratings',data=ratings,alpha=0.5)

# Now that we have a general idea of what the data looks like, let's move on to creating a simple recommendation system:
# Now let's create a matrix that has the user ids on one access and the game title on another axis. Each cell will then consist of the rating the user gave to that game 

gamemat = data.pivot_table(index='userId',columns='game',values='rating',fill_value=0)


# Normalisation

def norm_fun(row):
       new_row = (row - row.mean()) / (row.max() - row.min())
       return new_row
gamemat_std = gamemat.apply(norm_fun)

gamemat_std.head()

#  Most rated games:
ratings.sort_values('no_of_ratings',ascending=False).head(10)
ratings.head()

# Let's choose one game Rayman Origins which has the rating 5

rayman_user_ratings = gamemat_std['Rayman Origins']

rayman_user_ratings.head()

similar_to_rayman = gamemat_std.corrwith(rayman_user_ratings)

corr_rayman = pd.DataFrame(similar_to_rayman,columns=['Correlation'])
corr_rayman.dropna(inplace=True)
corr_rayman.head()

corr_rayman.sort_values('Correlation',ascending=False).head(10)

corr_rayman = corr_rayman.join(ratings['no_of_ratings'])
corr_rayman.head()
corr_rayman[corr_rayman['no_of_ratings']>4].sort_values('Correlation',ascending=False).head(1)

# now same for the other game

ratings.sort_values('no_of_ratings',ascending=False).head(10)
ratings.head()

# Let's choose one game Rayman Origins which has the rating 5
quake_user_ratings = gamemat_std['Quake']
quake_user_ratings.head()
similar_to_quake = gamemat_std.corrwith(quake_user_ratings)
corr_quake = pd.DataFrame(similar_to_quake,columns=['Correlation'])
corr_quake.dropna(inplace=True)
corr_quake.head()
corr_quake.sort_values('Correlation',ascending=False).head(10)
corr_quake = corr_quake.join(ratings['no_of_ratings'])
corr_quake.head()
corr_quake[corr_quake['no_of_ratings']>2].sort_values('Correlation',ascending=False).head(10)

##########################################################################################################

# Q2)


import pandas as pd

# import Dataset 
data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Recommendation Engine\Entertainment.csv", encoding = 'utf8')
data.shape # shape
data.columns
data.Category # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey-
#- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
data["Category"].isnull().sum() 

# there is no nan values

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(data.Category)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape 

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of data Titles  to index number 
data_index = pd.Series(data.index, index = data['Titles']).drop_duplicates()

Id = data_index["Pocahontas (1995)"]
Id
def get_recommendations(Name, topN):    
    topN = 10
    # Getting the movie index using its title 
    Id = data_index[Name]
    
    # Getting the pair wise similarity score for all the movies.  
    cosine_scores = list(enumerate(cosine_sim_matrix[Id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN + 1]
    
    # Getting the movie index 
    data_idx  =  [i[0] for i in cosine_scores_N]
    data_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    data_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    data_similar_show["Titles"] = data.loc[data_idx, "Titles"]
    data_similar_show["Score"] = data_scores
    data_similar_show.reset_index(inplace = True)  

    print (data_similar_show)
    # return(data_similar_show)

    
# Enter your Title and number of titles to be recommended 
get_recommendations("Pocahontas (1995)", topN = 10 )
data_index["Heat (1995)"]
`



