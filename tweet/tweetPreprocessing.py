import pandas as pd
import numpy as np
import os

from nltk.twitter.common import json2csv
from nltk.tokenize import TweetTokenizer
import torch


from tweet.textProc import pretokenization_cleaning
from tweet.normalizer import lemmatize
from tweet.postTokenCleaner import posttokenization_cleaning

from transformers import BertTokenizer, BertModel

def tokenize(text):
  tknzr = TweetTokenizer(reduce_len=True)
  return tknzr.tokenize(text)

def getEmbeddings(tweet_file):
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

  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  model = BertModel.from_pretrained('bert-base-uncased')

  tweets['BertEmbedding'] = [tokenizer.encode(' '.join(tokens), add_special_tokens=True, return_tensors='pt')  for tokens in tweets['postToken']]

  embeddings = []
  for elem in tweets['BertEmbedding']:
    with torch.no_grad():
        outputs = model(elem)
        last_hidden_states = outputs[0]
        embeddings.append(np.array(last_hidden_states.mean(dim=1).squeeze()))

  return embeddings
