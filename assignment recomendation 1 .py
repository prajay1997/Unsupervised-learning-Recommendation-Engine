# import os
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

# creating a mapping of anime name to index number 
data_index = pd.Series(data.index, index = data['Titles']).drop_duplicates()

Id = data_index["Pocahontas (1995)"]
Id
def get_recommendations(Name, topN):    
    topN = 5
    # Getting the movie index using its title 
    Id = data_index[Name]
    
    # Getting the pair wise similarity score for all the 
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
    # data_similar_show.drop(["index"], axis=1, inplace=True)
    print (data_similar_show)
    # return(data_similar_show)

    
# Enter your Title and number of titles to be recommended 
get_recommendations("Heat (1995)", topN = 5 )
data_index["Jumanji (1995)"]
`



