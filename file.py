import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
try:
    movies = pd.read_csv(r"C:\Users\arjun\OneDrive\Desktop\movies.csv")
except FileNotFoundError:
    print("Error: Unable to load the CSV file. Please check the file path.")
    exit()

# Select the features to use for generating recommendations
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace missing values with empty strings
for feature in selected_features:
    movies[feature] = movies[feature].fillna('')

# Combine all selected features into a single text
movies['combined_features'] = movies[selected_features].apply(lambda x: ' '.join(x), axis=1)

# Convert text into feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies['combined_features'])

# Calculate the cosine similarity score
similarity = cosine_similarity(feature_vectors)

# Get user input for a movie
movie_input = input("Enter a movie you like: ").strip().lower()
movie_titles = movies['title'].str.lower().tolist()

# Find the closest match for the input
matches = difflib.get_close_matches(movie_input, movie_titles)

if matches:
    match_title = matches[0]
    movie_index = movies[movies['title'].str.lower() == match_title].index[0]
    similarity_scores = list(enumerate(similarity[movie_index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\nMovies recommended based on '{movies.loc[movie_index, 'title']}':\n")

    try:
        count = int(input("How many recommendations do you want? "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        exit()

    i = 0
    for index, score in sorted_movies:
        if index != movie_index:
            print(movies.loc[index, 'title'])
            i += 1
        if i >= count:
            break
else:
    print("No matching movie found. Please try another name.")
