def remove_punc(list_token): 
  #print(list_token)
  def process(strg_token):
    strg_numb ='''0123456789'''
    strg_3dots='...'
    strg_2dots=".."
    strg_punc = '''!()+-[]{}|;:'"\\,<>./?@#$£%^&*_~“”…‘’'''
    strg_output=''
    #for idx, char in enumerate(strg_token): 
    #print(item)
    if (len(strg_token)==0): #empty char
        strg_output +=''
    else:
      if (all(char in strg_numb for char in strg_token) or
          strg_token[0] in strg_numb): #if char is a number
        strg_output +=''
      else:
        if (len(strg_token)==1 and strg_token in strg_punc): #if char is a single punc
          strg_output +=''
        else:
            if (strg_token[0]=='#'): #if char is hashtag
              strg_output +=strg_token.lower()
            elif(strg_token==strg_3dots or strg_token==strg_2dots):
              strg_output +=''
            else: # other than above, char could be part of word,
            # e.g key-in
              strg_output += strg_token
    return strg_output
  list_output=[process(token) for token in list_token]
  return list_output

def remove_empty_item(list_item):
  token = [token for token in list_item if len(token)>0]
  return token

def lowercase_alpha(list_token):
  return [token.lower() if (token.isalpha() or token[0]=='#') else token for token in list_token]

def strg_list_to_list(strg_list):
  return strg_list.strip("[]").replace("'","").replace('"',"").replace(",","").split()

def posttokenization_cleaning(unkn_input):
  list_output=[]
  if (isinstance(unkn_input,list)):
    list_output=unkn_input
  if (isinstance(unkn_input,str)):
    list_output=strg_list_to_list(unkn_input)
  list_output=remove_punc(list_output)
  list_output=remove_empty_item(list_output)
  #list_output=lowercase_alpha(list_output)


  return (list_output)