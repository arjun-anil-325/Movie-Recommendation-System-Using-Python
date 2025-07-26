# 🎬 Movie Recommendation System Using Python

## 📌 Project Overview
This project aims to build a **Movie Recommendation System** using Python and Machine Learning techniques. The system provides **personalized movie suggestions** to users, helping them discover films they are likely to enjoy.

## 🧠 Recommendation Techniques Used
The system combines **two major approaches**:

1. **Collaborative Filtering**
   - Learns from user interactions (like ratings).
   - Finds patterns among users with similar tastes.
   - Recommends movies based on similar users' preferences.

2. **Content-Based Filtering**
   - Uses movie features like genre, cast, and plot.
   - Recommends movies similar to those the user liked before.

By integrating both techniques, the system delivers **more accurate and personalized recommendations**.

## Features
- Simple text-based input and output
- Recommends similar movies using TF-IDF and cosine similarity
- Uses a dataset of movies with metadata
- User-friendly CLI interface

## How It Works
1. Loads movie data from a CSV file.
2. Combines relevant movie features into one text field.
3. Transforms this text using TF-IDF vectorization.
4. Calculates similarity between movies using cosine similarity.
5. Asks the user for a movie title and recommends similar ones.

## 🛠️ Technologies & Libraries
- **Python**
- **Pandas** – Data manipulation
- **Scikit-learn** – Machine learning models & similarity metrics
- **Surprise Library** – Collaborative filtering models
- **Difflib** – For matching close movie titles
- **TfidfVectorizer** – To analyze textual features of movies
- **Cosine Similarity** – To measure similarity between movies

## 📈 Model Evaluation
- The model performance is evaluated using:
  - **Root Mean Square Error (RMSE)**  
    To ensure reliable accuracy in recommendations.

## 🎯 Project Goals
- Deliver **relevant and engaging movie suggestions**.
- Simplify the movie discovery process for users.
- Show the power of **Machine Learning** in building real-world recommendation systems.
