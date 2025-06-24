
import pandas as pd

# Example lists
# list1 = [1, 2, 3, 4]
# list2 = ['a', 'b', 'c', 'd']
# list3 = [5,6,7,8]


# # Creating a DataFrame
# df = pd.DataFrame({'Column1': list1, 'Column2': list2})
# df['column3'] = list3
# # Saving the DataFrame to a CSV file
# df.to_csv('example.tsv',sep='\t', index=True)

# print("DataFrame saved to example.csv")



# import argparse
# def parcing():
#     parser  = argparse.ArgumentParser()
#     parser.add_argument("--out", type=str, default=f"./data/train")
#     args = parser.parse_args()
#     return args
# def main(args):
#     print(args.out)

# args = parcing()
# args.out = "hi"
# main(args)




# import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device(
#     "cuda" if torch.cuda.is_available()  else "cpu")
# n_gpu = torch.cuda.device_count()
# print(n_gpu , device)



# import torch

# print("CUDA available: ", torch.cuda.is_available())
# print("CUDA device count: ", torch.cuda.device_count())
# print("CUDA current device: ", torch.cuda.current_device())
# print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

# import pandas as pd
# a = pd.read_csv('./st_attack_pairs.tsv', sep='\t')
# b= pd.read_csv('./st_attack_pairs.tsv', sep='\t')
# a = pd.concat([a , b], axis=0)
# print(a)
# s = "hi my name is ()6tah'namy '"
# for i in s:
#     if not i.isalpha() and i != ' ' and not i.isdigit():
#         s = s.replace(i , '')

# print(s)

# import pandas as pd
# a = pd.read_csv("/b/tanmay/work/summer_project/prADA/total_monoT5_Reranked.tsv",sep='\t')
# b = a.loc[a['qid'] == 2]
# print(b)
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger_eng')

nltk.download('punkt_tab')

 
# // Dummy text
txt = "Sukanya, Rajib and Naba are my good friends. "
 
# sent_tokenize is one of instances of 
# PunktSentenceTokenizer from the nltk.tokenize.punkt module
 
# tokenized = sent_tokenize(txt)
# for i in tokenized:
     
#     # Word tokenizers is used to find the words 
#     # and punctuation in a string
#     wordsList = nltk.word_tokenize(i)
 
#     # removing stop words from wordList
#     wordsList = [w for w in wordsList if not w in stop_words] 
 
#     #  Using a Tagger. Which is part-of-speech 
#     # tagger or POS-tagger. 
#     tagged = nltk.pos_tag(wordsList)
 
#     print(tagged)

print(nltk.pos_tag(['doing']))



"""

from nltk.corpus import wordnet as wn

# Just to make it a bit more readable
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'


def convert(word, from_pos, to_pos):    
    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w:-w[1])

    # return all the possibilities sorted by probability
    print(f"{from_pos} -> {to_pos} : {word} -> {result}")
    return result


convert('direct', 'v', 'n')
convert('direct', 'a', 'n')
convert('quick', 'a', 'r')
convert('quickly', 'r', 'a')
convert('hunger', 'n', 'v')
convert('run', 'v', 'a')
convert('tired', 'a', 'r')
convert('tired', 'a', 'v')
convert('tired', 'a', 'n')
convert('tired', 'a', 's')
convert('wonder', 'v', 'n')
convert('wonder', 'n', 'a')

WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'


pos_dict = {'PRP$':'n',
             'VBG':'v',
             'FW': 'o',
             'VB':'v',
             'POS':'o',
             "''": '0',
             'VBP':'v',
             'VBN':'v',
             'JJ':'a',
             'WP':'o',
             'VBZ':'v', 
             'DT':'o',
             'RP':'o', 
             '$':'o', 
             'NN':'n', 
             ')':'o', 
             '(':'o', 
             'RBR':'r', 
             'VBD':'v', 
             ',':'o', 
             '.':'o', 
             'TO':'o', 
             'LS':'o', 
             'RB':'r', 
             ':':'o', 
             'NNS':'n', 
             'NNP':'n', 
             '``':'o', 
             'WRB':'r', 
             'CC':'o', 
             'PDT':'o', 
             'RBS':'r', 
             'PRP':'n', 
             'CD':'o', 
             'EX':'o', 
             'IN':'o', 
             'WP$':'n', 
             'MD':'v', 
             'NNPS':'n', 
             '--':'o', 
             'JJS':'a', 
             'JJR':'a', 
             'SYM':'o', 
             'UH':'o', 
             'WDT':'o'}
print(pos_dict)