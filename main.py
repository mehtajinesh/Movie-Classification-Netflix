"""Main file
"""
import sys
import pandas as pd
from preprocessing import PreprocessingData


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


def preprocess_movie_description(data_frame):
    """preprocess movie description
    """
    preprocess = PreprocessingData()
    text_column_dataframe = data_frame["description"]
    text_column_dataframe = preprocess.perform_lower_casing(
        text_column_dataframe)
    text_column_dataframe = preprocess.remove_extra_whitespaces(
        text_column_dataframe)
    text_column_dataframe = preprocess.perform_tokenization(
        text_column_dataframe)
    text_column_dataframe = preprocess.remove_stopwords(
        text_column_dataframe)
    text_column_dataframe = preprocess.remove_punctuations(
        text_column_dataframe)
    text_column_dataframe = preprocess.perform_lemmatize(
        text_column_dataframe)
    text_column_dataframe = preprocess.perform_stemming(
        text_column_dataframe)
    data_frame["description"] = text_column_dataframe
    return data_frame


def main(csv_file):
    """main function
    """
    # read data from csv file into numpy array
    raw_data_frame = read_data(csv_file)
    # fetch movie data which is free of duplicates
    movie_data_frame = fetch_cleaned_movie_data(raw_data_frame)
    # preprocess data (movie description)
    processed_data_frame = preprocess_movie_description(movie_data_frame)
    # We can perform clustering in a naive way using groupby as shown below.
    # However the clusters formed are not very good, in the sense that
    # they are not very homogeneous and miss a lot of context similarity.
    movies_grouped = processed_data_frame.groupby(
        ["title", "director", "listed_in", ]).apply(lambda df: df.title)
    print(movies_grouped.head())


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print('Usage: main.py <csv_file>')
    #     sys.exit(1)
    # main(sys.argv[1])
    main('data/netflix_titles.csv')
