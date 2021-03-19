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


def _print_unique_values_of_features(X):
    for col_name in X.columns:
        if X[col_name].dtypes == 'object':
            unique_cat = len(X[col_name].unique())
            print("Atribut '{col_name}' ima {unique_cat} jedinstvenih vrednosti".format(
                col_name=col_name, unique_cat=unique_cat))


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

def _remove_links(tweet):
    return re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", " ", tweet)

def _get_entities_list(tweets):
    entities_list = []
    url_list = []
    for index, row in tweets.iterrows():
        tweet = row['text']
        tweet = _remove_links(tweet)
        recognized_entities = nlp(tweet)
        # url_list_tmp=[]
        # for i, token in enumerate(recognized_entities):
        #     if token.like_url:
        #         token.tag_ = 'URL'
        #         url_list_tmp.append((str(token), "URL"))

        # url_list.append(url_list_tmp)

        entities_list.append([(X.text, X.label_)
                              for X in recognized_entities.ents])
    return entities_list, url_list


def _get_text_tags_list(tweets):
    topic_tags_list = []
    for index, row in tweets.iterrows():
        stop_words = set(stopwords.words('english'))
        tweet = row['text']
        tweet_letters = _remove_links(tweet)
        tweet_letters = re.sub("[^a-zA-Z]", " ", tweet_letters).lower()

        word_tokens = word_tokenize(tweet_letters)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        stemmed_words = []
        stemmer = PorterStemmer()
        #stemmer = WordNetLemmatizer()
        for word in filtered_sentence:
            word = stemmer.stem(word)
            #word = stemmer.lemmatize(word)
            stemmed_words.append(word)

        fdist = FreqDist(stemmed_words)
        most_frequent_words = fdist.most_common(5)
        topic_tags_list.append(most_frequent_words)
    return topic_tags_list


def _visualize_tweets(tweets, color, csv_name):
    most_common_number = 30

    filtered_text_list = _get_text_tags_list(tweets)
    entities_list, url_list = _get_entities_list(tweets)

    tweets = tweets.assign(filtered_text=filtered_text_list)
    tweets = tweets.assign(entities=entities_list)
    #tweets = tweets.assign(urls=url_list)

    # all_url_entities = []
    # for row in tweets['urls']:
    #     if len(row) != 0:
    #         for pair in row:
    #             if pair[0]:
    #                 all_url_entities.append(pair[0])

    # all_words_counted = Counter(all_url_entities).most_common(most_common_number)
    # all_words, all_words_count = _get_words_and_word_count(all_words_counted)
    # _plot_horizontal_bar_chart(
    #     all_words, all_words_count, color, "Most frequent url entities")

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

    """ Load csv file """
    filename = 'twitter.csv'
    df = _load_csv(filename)

    """ Select the names of the columns to be dropped """
    labels_to_drop = ['in_reply_to_screen_name',
                      'in_reply_to_status_id',
                      'in_reply_to_user_id',
                      'is_quote_status',
                      'longitude',
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
                      'entities',
                      'time',
                      'retweet_count',
                      'favorite_count',
                      'extended_entities']

    pprint(list(df.columns.values))

    """ Drop the selected columns """
    df = df.drop(axis=1, labels=labels_to_drop)
    pprint(list(df.columns.values))
    pprint(df.head(5))

    """ Print the unique values of columns """
    _print_unique_values_of_features(df)

    """ Look the values for land, handle and original_author """
    pprint(df['lang'].value_counts())
    pprint(df['handle'].value_counts())
    pprint(df['original_author'].value_counts())

    pprint("----- Check for missing values -----")
    pprint(df.isnull().sum())

    """ Delete all rows that have non english language """
    df = df[df.lang == 'en']
    pprint(df['lang'].value_counts())

    """ Split dataset into original tweets and reteets """
    df_original = df[df.original_author.isnull() == True]
    df_retweets = df[df.original_author.isnull() == False]

    pprint(df_original.head(5))

    """ Split original tweets into hilarys and donalds """
    donald_tweets = df_original[df_original.handle == 'realDonaldTrump']
    hilary_tweets = df_original[df_original.handle == 'HillaryClinton']

    """ Split retweeted tweets into hilarys and donalds """
    donald_retweets = df_retweets[df_retweets.handle == 'realDonaldTrump']
    hilary_retweets = df_retweets[df_retweets.handle == 'HillaryClinton']

    pprint(donald_tweets['handle'].value_counts())
    pprint(hilary_tweets['handle'].value_counts())

    _visualize_tweets(donald_tweets, "teal", "tweets_donald")
    _visualize_tweets(hilary_tweets, "deeppink", "tweets_hilary")

    _visualize_tweets(donald_retweets, "deepskyblue", "retweets_donald")
    _visualize_tweets(hilary_retweets, "orchid", "retweets_hilary")

    pprint("------- Donald Trump retweets -------------------")
    pprint(donald_retweets['original_author'].value_counts())
    pprint("-------Hilary Clinton retweets -------------------")
    pprint(hilary_retweets['original_author'].value_counts())

    # donald_tweets.to_csv("./tweets_donald.csv", index=False, header=True)
    # hilary_tweets.to_csv("./tweets_hilary.csv", index=False, header=True, encoding='utf-8-sig')
    # donald_retweets.to_csv("./retweets_donald.csv", index=False, header=True)
    # hilary_retweets.to_csv("./retweets_hilary.csv", index=False, header=True)
    # df.to_csv("./tweets_cleaned.csv", index=False, header=True)
