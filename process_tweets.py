import pandas as pd
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
nlp = en_core_web_sm.load()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def _load_csv(filename):
    df = pd.read_csv(filename, na_values=['#NAN'])
    return df


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

    filtered_text_list = []
    entities_list = []

    for index, row in donald_tweets.iterrows():
        stop_words = set(stopwords.words('english'))
        sent = row['text']
        doc = nlp(sent)
        sent_letters=re.sub("[^a-zA-Z]", " ", sent).lower()
        word_tokens = word_tokenize(sent)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        stemmed_words = []
        stemmer = WordNetLemmatizer()
        for word in filtered_sentence:
            word = stemmer.lemmatize(word)
            stemmed_words.append(word)
        entities_list.append([(X.text, X.label_) for X in doc.ents])
        filtered_text_list.append(stemmed_words)

    donald_tweets = donald_tweets.assign(filtered_text=filtered_text_list)
    donald_tweets = donald_tweets.assign(entities=entities_list)

    pprint(donald_tweets['filtered_text'].head(10))
    pprint(donald_tweets['entities'].head(10))

    all_entities=[]
    for row in donald_tweets['entities']:
        if len(row) != 0:
            for pair in row:
                if pair[0]:
                    all_entities.append(pair[0])
    
    pprint(Counter(all_entities).most_common(10))

    all_person_entities=[]
    for row in donald_tweets['entities']:
        if len(row) != 0:
            for pair in row:
                if pair[1]:
                     if pair[1]=='PERSON':
                        all_person_entities.append(pair[0])
    
    pprint(Counter(all_person_entities).most_common(20))


    non_person_entities=[]
    for row in donald_tweets['entities']:
        if len(row) != 0:
            for pair in row:
                if pair[1]:
                     if pair[1]!='PERSON':
                        non_person_entities.append(pair[0])
    
    pprint(Counter(non_person_entities).most_common(20))



