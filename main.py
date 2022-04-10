"""Main file
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize

stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")


def read_data(csv_file):
    """read data from file
    """
    # extract movie data from dataset into pandas dataframe
    # return data
    data_frame = pd.read_csv(csv_file, sep=',')
    return data_frame


def fetch_cleaned_movie_data(raw_data_frame):
    """data cleaning and fetching movie data

    Args:
        raw_data_frame (dataframe): raw dataframe from csv file
    """
    # remove duplicates
    raw_data_frame.drop_duplicates(subset=['title'], inplace=True)
    # select only movie data
    data_frame = raw_data_frame[raw_data_frame['type'] == 'Movie']
    return data_frame


def tokenize_and_stem(text):
    """tokenize and stem the text

    Args:
        text (str): sentence

    Returns:
        list: stemmed sentence
    """
    tokens = [word for sent in sent_tokenize(
        text) for word in word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    # exclude stopwords from stemmed words
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopwords]
    return stems


def perform_vectorization(transformed_data):
    """perform vectorization using tf-idf

    Args:
        transformed_data (array): transformed data (movie description)

    Returns:
        tuple: tf-idf vectorized data, tf-idf vectorizer
    """
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_stem, strip_accents='ascii', lowercase=True, use_idf=True,
        norm=u'l2', smooth_idf=True)
    tfidf_transformed_data = tfidf_vectorizer.fit_transform(transformed_data)
    return tfidf_transformed_data, tfidf_vectorizer


def clustering_errors(k, data):
    """calculate clustering errors

    Args:
        k (int): number of clusters
        data (list of str): list of movie description

    Returns:
        int: silhouette score
    """
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    #cluster_centers = kmeans.cluster_centers_
    # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values, predictions)]
    # return sum(errors)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg


def identify_best_cluster_value(data):
    """We will use the elbow method to find the best k value.
    """
    # Choose the range of k values to test.
    # We added a stride of 5 to improve performance.
    # We don't need to calculate the error for every k value
    possible_k_values = range(2, 100, 5)
    # Define function to calculate the clustering errors

    # Calculate error values for all k values we're interested in
    errors_per_k = [clustering_errors(k, data) for k in possible_k_values]
    # Plot the each value of K vs. the silhouette score at that value
    fig, axis = plt.subplots(figsize=(16, 6))
    plt.plot(possible_k_values, errors_per_k)
    # Ticks and grid
    xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
    axis.set_xticks(xticks, minor=False)
    axis.set_xticks(xticks, minor=True)
    axis.xaxis.grid(True, which='both')
    yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
    axis.set_yticks(yticks, minor=False)
    axis.set_yticks(yticks, minor=True)
    axis.yaxis.grid(True, which='both')
    return -1


def perform_k_means_mini_batch(td_idf_transformed_data, vector, n_clusters, movie_data_description):
    """perform k-means clustering

    Args:
        transformed_data (array): transformed data (movie description)
    """
    kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++')
    kmeans_model.fit(td_idf_transformed_data)
    request_transform = vector.transform(movie_data_description)
    data_frame_cluster_predictions = kmeans_model.predict(request_transform)
    return data_frame_cluster_predictions


def main(csv_file):
    """main function
    """
    # read data from csv file into numpy array
    raw_data_frame = read_data(csv_file)
    # fetch movie data which is free of duplicates
    movie_data_frame = fetch_cleaned_movie_data(raw_data_frame)
    # We can perform clustering in a naive way using groupby as shown below.
    # However the clusters formed are not very good, in the sense that
    # they are not very homogeneous and miss a lot of context similarity.
    # movies_grouped = movie_data_frame.groupby(
    #     ["title", "director", "listed_in", ]).apply(lambda df: df.title)
    # print(movies_grouped.head())

    # Since the basic approach wont work, we will use movie description
    # to perform clustering.
    tf_idf_transformed_data, vector = perform_vectorization(
        movie_data_frame["description"])
    #ideal_n_cluster = identify_best_cluster_value(tf_idf_transformed_data)
    # identified this value of 37 using elbow method
    ideal_n_cluster = 37
    clustered_labels = perform_k_means_mini_batch(
        tf_idf_transformed_data, vector, ideal_n_cluster, movie_data_frame["description"])
    movie_data_frame_labelled = movie_data_frame.assign(
        cluster=clustered_labels)


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print('Usage: main.py <csv_file>')
    #     sys.exit(1)
    # main(sys.argv[1])
    main('data/netflix_titles.csv')
