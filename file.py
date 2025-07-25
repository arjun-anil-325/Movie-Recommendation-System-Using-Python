import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv(r"C:\Users\arjun\OneDrive\Desktop\movies.csv") # loading the data from the csv file to apandas dataframe

selected_features = ['genres','keywords','tagline','cast','director'] # selecting the relevant features for recommendation

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('') # replacing the null valuess with null string

# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']# combining all the 5 selected features # combining all the 5 selected features# co

# print(combined_features)

vectorizer = TfidfVectorizer() # converting the text data to feature vectors
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)

similarity = cosine_similarity(feature_vectors) # getting the similarity scores using cosine similarity
# print(similarity)

movie_name = input(' Enter your favourite movie name : ') # getting the movie name from the user
list_of_all_titles = movies_data['title'].tolist() # creating a list with all the movie names given in the dataset
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles) # finding the close match for the movie name given by the user
# print(find_close_match)

close_match = find_close_match[0]
# print(close_match)

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0] # finding the index of the movie with title
# print(index_of_the_movie)

similarity_score = list(enumerate(similarity[index_of_the_movie])) # getting a list of similar movies
# print(similarity_score)

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) # sorting the movies based on their similarity score
# print(sorted_similar_movies)

print('Movies suggested for you : \n') # print the name of similar movies based on the index

i = 1
required= int(input("Enter the number of movies required : "))
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<required+1):
    print(i, '.',title_from_index)
    i+=1