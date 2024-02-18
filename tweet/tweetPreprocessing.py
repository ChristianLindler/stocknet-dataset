import pandas as pd
import os

from nltk.twitter.common import json2csv
from nltk.tokenize import TweetTokenizer


from tweet.textProc import pretokenization_cleaning
from tweet.normalizer import lemmatize
from tweet.postTokenCleaner import posttokenization_cleaning



def tokenize(text):
  tknzr = TweetTokenizer(reduce_len=True)
  return tknzr.tokenize(text)

def getTokens(tweet_file, tokenizer):
  with open(tweet_file) as fp:
      json2csv(fp, 'temp.csv', ['created_at', 'text'])

  tweets = pd.read_csv('temp.csv', encoding='utf-8')
  os.remove('temp.csv')

  tweets['created_at'] = pd.to_datetime(tweets['created_at'], format='%a %b %d %H:%M:%S %z %Y')
  tweets = tweets.sort_values(by='created_at')

  tweets['pretoken'] = [pretokenization_cleaning(sentence) for sentence in tweets['text']]
  tweets['token'] = [tokenize(pretoken) for pretoken in tweets['pretoken']]
  tweets['normalized'] = [lemmatize(token) for token in tweets['token']]
  tweets['postToken'] = [posttokenization_cleaning(normalized) for normalized in tweets['normalized']]
  tweets['BertEmbedding'] = [tokenizer.encode(tokens) for tokens in tweets['postToken']]

  #import csv
  #tweets.to_csv('testing.csv',columns=tweets.columns.values,quoting=csv.QUOTE_ALL, index=False)
  max_length = 15
  padded_embeddings = [embedding[:max_length] + [0] * max(0, max_length - len(embedding)) for embedding in tweets['BertEmbedding']]

  return padded_embeddings