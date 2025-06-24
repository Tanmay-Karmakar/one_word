from transformers import BertTokenizer
import numpy as np
import pandas as pd

def get_q_center_vec(query, counter_fitted_vec_path, tokenizer): # ,query_tok = None
    # print(f"query = {query}\n type = {type(query)}")
    tokens = tokenizer.tokenize(query)
    n = len(tokens)
    vectors = []
    word_found = 0
    with open(counter_fitted_vec_path, 'r') as sef:
        for line in sef:
            l = line.split()
            word = l[0]
            
            if str(word) in tokens:
                # print(word)
                word_found += 1
                vec = l[1:]
                vectors.append(vec)
            
            if word_found == n:
                break
            
    vectors = list(map(lambda sublist: list(map(float, sublist)), vectors))
    sum = vectors[0]
    for i in range(1,len(vectors)):
        sum = np.add(sum, vectors[i]).tolist()
        
    center_vec = list(map(lambda num: num / len(vectors), sum))
    return center_vec


def compute_sim(vec1, vec2):
    vec1 =  np.array(vec1,dtype=float)
    vec2 = np.array(vec2,dtype=float)
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosine_sim


def find_close_word(embd_path, vec, candi_num=5000000, sim_threshold = 0):
    candi_words = {}
    num_word = 0
    with open(embd_path, 'r') as sef:
        for line in sef:
            l = line.split()
            word = l[0]
            vector = l[1:]
            sim = compute_sim(vec, vector)
            # print(f"word = {word}, Sim = {sim}")
            if sim >= sim_threshold:
                candi_words[word] = sim
                num_word+= 1
            if num_word >= candi_num:
                break
    if len(candi_words) == 0:
        print("F")
        return -1
    res = max(candi_words, key=candi_words.get)
    return res, candi_words[res]

def get_query_center(query, counter_fitted_vec_path, tokenizer):
    center_vec = get_q_center_vec(query, counter_fitted_vec_path, tokenizer)
    candi_word, sim = find_close_word(counter_fitted_vec_path, center_vec)
    return candi_word, sim



if __name__=='__main__':
    cfvp = './models/counter-fitted-vectors.txt'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    s = 2
    if s == 1:
        query = ''
        if query == None or query == '':
            query = input("Enter the query: \n")
        center, sim = get_query_center(query, cfvp, tokenizer)
        print(f"query = {query}\nquery center = {center}\nsimilarity = {sim}")
    else:
        dataset = 'trec_dl_2020'
        save_file = f"./query_centers/{dataset}.tsv"
        data_file = dataset.replace('_','-')
        query_path = f'./data/msmarco-passage/{data_file}/queries.tsv'
        
        col = ['qid', 'query']
        queries = pd.read_csv(query_path, sep='\t',header = None, names = col)
        centers = []
        sims = []
        i = 1
        for query in queries['query']:
            center, sim = get_query_center(query, cfvp, tokenizer)
            print(f"{i}. query = {query}\n\t center = {center}\n\t sim = {sim}\n")
            centers.append(center)
            sims.append(sim)
            i += 1
        queries['center'] = centers
        queries['sim'] = sims
        queries.to_csv(save_file, sep='\t', index = False)
    
    
            
        
        