"""Main file
"""
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # errors = [mean_squared_error(row, cluster_centers[cluster])
    # for row, cluster in zip(data.values, predictions)]
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


def f1_score_func(preds, labels):
    """_summary_

    Args:
        preds (_type_): _description_
        labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels):
    """_summary_

    Args:
        preds (_type_): _description_
        labels (_type_): _description_
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


def evaluate(model, dataloader_val):
    """_summary_

    Args:
        model (_type_): _description_
        dataloader_val (_type_): _description_

    Returns:
        _type_: _description_
    """

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


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
    ideal_n_cluster = 10
    clustered_labels = perform_k_means_mini_batch(
        tf_idf_transformed_data, vector, ideal_n_cluster, movie_data_frame["description"])
    movie_data_frame_labelled = movie_data_frame.assign(
        cluster=clustered_labels)
    # print(movie_data_frame_labelled['cluster'].value_counts())
    # print(movie_data_frame_labelled.cluster.values)
    # print(movie_data_frame_labelled.cluster.values.shape)
    # print(movie_data_frame_labelled.description.values)
    # print(movie_data_frame_labelled.description.values.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        movie_data_frame_labelled.description.values,
        movie_data_frame_labelled.cluster.values,
        test_size=0.15,
        random_state=42, stratify=movie_data_frame_labelled.cluster.values)

    movie_data_frame_labelled['data_type'] = [
        'not_set']*movie_data_frame_labelled.shape[0]

    for item in movie_data_frame_labelled.description:
        if item in X_train:
            movie_data_frame_labelled.loc[movie_data_frame_labelled.description ==
                                          item, 'data_type'] = 'train'
        else:
            movie_data_frame_labelled.loc[movie_data_frame_labelled.description ==
                                          item, 'data_type'] = 'val'

    # print(movie_data_frame_labelled.groupby(
    #     ['cluster', 'data_type']).count())
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        movie_data_frame_labelled[movie_data_frame_labelled.data_type ==
                                  'train'].description.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        movie_data_frame_labelled[movie_data_frame_labelled.data_type ==
                                  'val'].description.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(
        movie_data_frame_labelled[movie_data_frame_labelled.data_type ==
                                  'train'].cluster.values).type(torch.LongTensor)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(
        movie_data_frame_labelled[movie_data_frame_labelled.data_type == 'val'].cluster.values).type(torch.LongTensor)

    dataset_train = TensorDataset(
        input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(
            movie_data_frame_labelled['cluster'].unique()),
        output_attentions=False,
        output_hidden_states=False)
    batch_size = 3

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)
    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)

    epochs = 5

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    for epoch in tqdm(range(1, epochs+1)):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(
            epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(
                {'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        torch.save(model.state_dict(),
                   f'data_volume/finetuned_BERT_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(
            model, dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(
                movie_data_frame_labelled['cluster'].unique()),
            output_attentions=False,
            output_hidden_states=False)

    model.to(device)
    model.load_state_dict(torch.load(
        'data_volume/finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))

    _, predictions, true_vals = evaluate(model, dataloader_validation)
    accuracy_per_class(predictions, true_vals)


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print('Usage: main.py <csv_file>')
    #     sys.exit(1)
    # main(sys.argv[1])
    main('data/netflix_titles.csv')
