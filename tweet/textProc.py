import re
import contractions

# stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_retweet_label(text):
  return re.sub('RT @[\w_]+:','', text)

def remove_video_label(text):
  return re.sub('VIDEO:','', text)

def remove_hyperlink(text):
  return re.sub(r'http\S+','', text) # r=raw \S=string

def remove_twitterhandle(text):
  return re.sub('@[A-Za-z0-9_]+(:)?','', text)

def remove_escape_sequence(text):
  return re.sub(r'\n','', text)

def remove_extra_spaces(text):
  return  re.sub(r"\s+"," ", text)  

def remove_contraction(text):
  arr = []
  for word in text.split():
      try:
        arr.append(contractions.fix(word))
      except:
        arr.append(word)
  text = ' '.join(arr)
  return text
  
def remove_stopwords(text):
  return " ".join([word for word in text.split() if word not in stop_words])

def pretokenization_cleaning(text):
  text=remove_retweet_label(text)
  text=remove_video_label(text)
  text=remove_hyperlink(text)
  text=remove_twitterhandle(text)
  text=remove_escape_sequence(text)
  text=remove_extra_spaces(text)  
  text=remove_contraction(text)
  text=remove_stopwords(text)
  return text