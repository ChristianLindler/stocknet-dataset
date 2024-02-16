#Normalizing task using Lemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def strg_list_to_list(strg_list):
  return strg_list.strip("[]").replace("'","").replace('"',"").replace(",","").split() 

#lemmatize requires list input
def lemmatize(unkn_input):
    if (isinstance(unkn_input,list)):
      list_input=unkn_input
    if (isinstance(unkn_input,str)):
      list_input=strg_list_to_list(unkn_input)
    list_sentence = [item.lower() for item in list_input]
    nltk_tagged = nltk.pos_tag(list_sentence)  
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])),nltk_tagged)
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        #" ".join(lemmatized_sentence)
    return lemmatized_sentence