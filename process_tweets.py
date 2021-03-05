import pandas as pd
import numpy as np
from pprint import pprint
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import nltk
import re
from collections import Counter
from nltk.chunk import conlltags2tree, tree2conlltags
import spacy
from spacy import displacy
import en_core_web_sm
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

nlp = en_core_web_sm.load()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def _load_csv(filename):
    df = pd.read_csv(filename, na_values=['#NAN'])
    return df


def _plot_horizontal_bar_chart(all_words, all_words_count, color, title):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(all_words))
    x = np.array(all_words_count)
    my_yticks = all_words
    ax.set_yticks(y_pos)
    ax.set_yticklabels(my_yticks)
    ax.barh(y_pos, x, align='center', color=color)
    ax.invert_yaxis()
    ax.set_title(title)
    plt.show()


def _get_words_and_word_count(all_words_counted):
    all_words = []
    all_words_count = []
    for pair in all_words_counted:
        all_words.append(pair[0])
        all_words_count.append(pair[1])

    return all_words, all_words_count


def _get_entities_list(tweets):
    entities_list = []
    for index, row in tweets.iterrows():
        sent = row['text']
        doc = nlp(sent)
        entities_list.append([(X.text, X.label_) for X in doc.ents])
    return entities_list


def _get_filtered_text_list(tweets):
    filtered_text_list = []
    for index, row in tweets.iterrows():
        stop_words = set(stopwords.words('english'))
        sent = row['text']
        sent_letters = re.sub("[^a-zA-Z]", " ", sent).lower()
        word_tokens = word_tokenize(sent_letters)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        stemmed_words = []
        stemmer = WordNetLemmatizer()
        for word in filtered_sentence:
            word = stemmer.lemmatize(word)
            stemmed_words.append(word)

        fdist = FreqDist(stemmed_words)
        most_frequent_words = fdist.most_common(5)
        filtered_text_list.append(most_frequent_words)
    return filtered_text_list


def _visualize_tweets(tweets, color):
    most_common_number = 30

    filtered_text_list = _get_filtered_text_list(tweets)
    entities_list = _get_entities_list(tweets)

    tweets = tweets.assign(filtered_text=filtered_text_list)
    tweets = tweets.assign(entities=entities_list)

    pprint(tweets['filtered_text'].head(10))
    pprint(tweets['entities'].head(10))

    all_entities = []
    for row in tweets['entities']:
        if len(row) != 0:
            for pair in row:
                if pair[0]:
                    all_entities.append(pair[0])

    all_words_counted = Counter(all_entities).most_common(most_common_number)
    all_words, all_words_count = _get_words_and_word_count(all_words_counted)
    _plot_horizontal_bar_chart(
        all_words, all_words_count, color, "Most frequent entities")

    all_person_entities = []
    for row in tweets['entities']:
        if len(row) != 0:
            for pair in row:
                if pair[1]:
                    if pair[1] == 'PERSON':
                        all_person_entities.append(pair[0])

    all_words_counted = Counter(
        all_person_entities).most_common(most_common_number)
    all_words, all_words_count = _get_words_and_word_count(all_words_counted)
    _plot_horizontal_bar_chart(
        all_words, all_words_count, color, "Most frequent person entities")

    non_person_entities = []
    for row in tweets['entities']:
        if len(row) != 0:
            for pair in row:
                if pair[1]:
                    if pair[1] != 'PERSON':
                        non_person_entities.append(pair[0])

    all_words_counted = Counter(
        non_person_entities).most_common(most_common_number)
    all_words, all_words_count = _get_words_and_word_count(all_words_counted)
    _plot_horizontal_bar_chart(
        all_words, all_words_count, color, "Most frequent non person entities")

    all_filtered_words = []
    for row in tweets['filtered_text']:
        if len(row) != 0:
            for pair in row:
                if pair[1]:
                    all_filtered_words.append(pair[0])

    all_words_counted = Counter(
        all_filtered_words).most_common(most_common_number)
    all_words, all_words_count = _get_words_and_word_count(all_words_counted)
    _plot_horizontal_bar_chart(
        all_words, all_words_count, color, "Most frequent filtered words")


if __name__ == "__main__":

    filename = 'tweets.csv'
    df = _load_csv(filename)

    labels_to_drop = ['in_reply_to_screen_name',
                      'in_reply_to_status_id',
                      'in_reply_to_user_id',
                      'is_quote_status', 'longitude',
                      'latitude',
                      'place_id',
                      'place_full_name',
                      'place_name',
                      'place_type',
                      'place_country_code',
                      'place_country',
                      'place_contained_within',
                      'place_attributes',
                      'place_bounding_box',
                      'source_url',
                      'truncated',
                      'original_author',
                      'entities',
                      'extended_entities']
    pprint(list(df.columns.values))
    df = df.drop(axis=1, labels=labels_to_drop)
    pprint(list(df.columns.values))
    pprint(df.head(5))
    pprint(df['lang'].value_counts())
    pprint(df['handle'].value_counts())

    pprint("----- Check for missing values -----")
    pprint(df.isnull().sum())

    """ Delete all rows that have non english language """
    df = df[df.lang == 'en']
    pprint(df['lang'].value_counts())

    #df.to_csv("./tweets_cleaned.csv", index=False, header=True)
    pprint(df.head(5))

    donald_tweets = df[df.handle == 'realDonaldTrump']
    hilary_tweets = df[df.handle == 'HillaryClinton']
    pprint(donald_tweets['handle'].value_counts())
    pprint(hilary_tweets['handle'].value_counts())

    _visualize_tweets(donald_tweets, "blue")

    _visualize_tweets(hilary_tweets, "deeppink")
