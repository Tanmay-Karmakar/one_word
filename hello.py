import pandas as pd
from transformers import BertTokenizer
from get_query_center import get_query_center
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
qid = 1037798
col = ['qid', 'query']
query_file = pd.read_csv("./data/msmarco-passage/trec-dl-2019/queries.tsv",sep='\t',header = None, names=col)
query_text = query_file[query_file['qid']==qid]['query'].iloc[0]
print(query_text, type(query_text),'\n----')
query_center, sim = get_query_center('who is robert gray', './models/counter-fitted-vectors.txt', tokenizer)
print(query_center, sim)