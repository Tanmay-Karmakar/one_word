from reranker import RerankerForInference
# rk = RerankerForInference.from_pretrained("./models/Luyu-bert-base-mdoc-bm25")  # load checkpoint


import pandas as pd

import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'

import pyterrier as pt
from pyterrier.datasets import get_dataset, Dataset
import ir_datasets

if not pt.started():
    pt.init()

import time

start = time.time()


# print(top100_by_bm25)
# dataset = get_dataset("msmarco_passage")
# corpus_iter = dataset.get_corpus()
# print(type(corpus_iter))
# corpus = pd.DataFrame(corpus_iter[0], columns=["docno", "text"])

# def get_document_text(doc_id):
#     doc = corpus[corpus['docno'] == doc_id]
#     if not doc.empty:
#         return doc['text'].values[0]
#     else:
#         return None

# doc_id = "1234567"  # Replace with your actual document ID
# doc_text = get_document_text(doc_id)
# print(doc_text)

from transformers import T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

# num_special_tokens = 3
# # Model has 3 special tokens which take up the input ids 0,1,2 of ByT5.
# # => Need to shift utf-8 character encodings by 3 before passing ids to model.

# input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens

# labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

# loss = model(input_ids, labels=labels).loss
# loss.item()


def get_score_char(query, doc):
    num_special_tokens = 3
    input_ids = torch.tensor([list(query.encode("utf-8"))])+num_special_tokens
    labels = torch.tensor([list(doc.encode("utf-8"))])+num_special_tokens
    loss= model(input_ids, labels=labels).loss
    # print(f"hi {loss.item()}")
    return -loss.item()


dataset = ir_datasets.load("msmarco-passage")

def reranker(query , topk):
    if topk > 1000:
        topk = 1000
    bm25 = pt.BatchRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
    topk_by_bm25 = bm25.search(query)[:topk]
    score_list = []
    docid_list = []
    j=0
    for i in topk_by_bm25["docid"]:
        print(i)
        for doc in dataset.docs_iter():
            
            if doc.doc_id == str(i):
                text = doc.text
                print(j)
                j+=1
                break

        # inputs = rk.tokenize(query, text, return_tensors='pt')
        # score = float(rk(inputs).logits)
        score = get_score_char(query, text)
        docid_list.append(str(i))
        score_list.append(score)
        
    rank = list(range(0,len(docid_list)))
    
    docid_rscore = pd.DataFrame({'docid': docid_list , 'rerank_score': score_list , 'rank': rank})
    docid_rscore = docid_rscore.sort_values(by='rerank_score', ascending=False)
    docid_rscore['rank'] = rank
    return docid_rscore

# r = reranker('who is john moody', 100)  # 1034453

# r.to_csv('rerank_score_rank.tsv',sep='\t', index=False)

# Load TREC DL 2019 queries
previous_done = 34
queries = pd.read_csv("./data/msmarco-passage/trec-dl-2019/queries.tsv", sep="\t", names=["qid", "query"])
qqq = 0
for index, row in queries.iterrows():
    qqq+=1
    if qqq<= previous_done:
        print(f"done {qqq}")
        continue
    qid = row["qid"]
    query = row["query"]
    print(f"Processing query ({qqq}):{qid}: {query}")
    
    r = reranker(query, 100)
    with open(f"char_rerank.tsv", "a") as f:
        for i, row in r.iterrows():
            f.write(f"{qid}\tQ0\t{row['docid']}\t{row['rank']}\t{row['rerank_score']}\tgoogle-byt5-small\n")


end = time.time()
print(f"Total time taken = {end - start}")
        
