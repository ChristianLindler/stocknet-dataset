import pandas as pd
import os
from nltk.twitter.common import json2csv
from nltk.tokenize import TweetTokenizer
from textProc import pretokenization_cleaning
from normalizer import lemmatize
from postTokenCleaner import posttokenization_cleaning

def tokenize(text):
  tknzr = TweetTokenizer(reduce_len=True)
  return tknzr.tokenize(text)

def getTokens(tweet_file):
  with open(tweet_file) as fp:
      json2csv(fp, 'temp.csv', ['created_at', 'text'])

  tweets = pd.read_csv('temp.csv')
  os.remove('temp.csv')

  ##TODO: how should data points from the wrong day be dealt with?
  tweets.sort_values(['created_at'])

  tweets['pretoken'] = [pretokenization_cleaning(sentence) for sentence in tweets['text']]
  tweets['token'] = [tokenize(pretoken) for pretoken in tweets['pretoken']]
  tweets['normalized'] = [lemmatize(token) for token in tweets['token']]
  tweets['postToken'] = [posttokenization_cleaning(normalized) for normalized in tweets['normalized']]

  return tweets['postToken'].tolist()
  #tweets.to_csv('testing.csv',columns=tweets.columns.values,quoting=csv.QUOTE_ALL, index=False)