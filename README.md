# Movie-Classification-Netflix

This project tries to build a classification model and generate cluster labels using Netflix Dataset available on the kaggle platform.

## Dataset

For this project we consider the Netflix Title Dataset present here:[Netflix Dataset](https://www.kaggle.com/shivamb/netflix-shows?select=netflix_titles.csv)

## Steps for forming clusters

1. Download the dataset from the given location in Dataset Section.
2. Extract the csv file from the archive file and place in the current working directory under 'data' folder.
3. Preprocessing Data - stemming and removing stop words
4. Vectorize using TFIDF
5. Identify ideal clusters using elbow method
6. Form clusters using K-Means (mini-batch mode)

## Built with Tools

1. Python 3.8.10 (64-bit)
2. NLTK
3. Sklearn