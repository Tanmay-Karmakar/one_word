import logging, argparse, os, time, torch
# from myutils.word_recover.Bert_word_recover import BERTWordRecover
from reranker import RerankerForInference
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
from sentence_transformers import SentenceTransformer
from transformers import (BertConfig, BertTokenizer)
import transformers
import pandas as pd
transformers.logging.set_verbosity(transformers.logging.ERROR)
import tensorflow as tf
import tensorflow_hub as hub
from get_query_center import get_query_center
from sentence_transformers import util
import time

print("******************Hi****************")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("@@@@@@@@@@@@@@@bye@@@@@@@@@@@@@@@@@")
# clean_embeddings = embed(texts)

# adv_embeddings = embed(adv)
# cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))


# word_re = BERTWordRecover(embed_name=None, bert_tokenizer, bert_vocab_path="hi", max_query_length=50, max_pass_length=100)

def get_score_monoT5(query, pas, model_ref):
    """
    Gets the score of a passage for a query using the monoT5 model.

    Args:
        query (str): The query string.
        pas (str): The passage string.
        model_ref (object): The monoT5 model reference.

    Returns:
        float: The score of the passage.
    """
    df = pd.DataFrame([['q1', query, 'd1', pas]], columns=['qid', 'query', 'docno', 'text'])
    output_df = model_ref.transform(df)
    score = output_df['score'][0]
    return score

def get_score_luyu(query, pas, model_ref):
    """
    Gets the score of a passage using the Luyu-bert-base model.

    Args:
        query (str): The query string.
        pas (str): The passage string.
        model_ref (object): The Luyu model reference.

    Returns:
        float: The score of the passage.

    """
    inputs = model_ref.tokenize(query, pas, return_tensors='pt')
    score = float(model_ref(inputs).logits) 
    return score


def get_score_distilbert(query, pas, model_ref):
    """
    Gets the score of a passage using the distilbert model.

    Args:
        query (str): The query string.
        pas (str): The passage string.
        model_ref (object): The Luyu model reference.

    Returns:
        float: The score of the passage.

    """
    query_emb = model_ref.encode(query)
    doc_emb = model_ref.encode(pas)
    score = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()[0]
    
    return score

def get_score_model(query, pas, model_ref, model_name):
    if model_name == 'monoT5':
        return get_score_monoT5(query, pas, model_ref)
    elif model_name.lower() == 'luyu':
        return get_score_luyu(query, pas, model_ref)
    elif model_name == 'distilbert':
        return get_score_distilbert(query, pas, model_ref)
    else:
        raise ValueError('Invalid model name')
    
def find_sim(qc, word):
    qc_embedding = embed([qc])
    word_embedding = embed([word])
    cosine_sim = tf.reduce_mean(tf.reduce_sum(qc_embedding * word_embedding, axis=1))
    # print(cosine_sim)
    return float(cosine_sim)

def find_qc_sim_indices(qc, tokens):
    ind_sim_dict = {}
    n = len(tokens)
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            ind_sim_dict[i]=0
        elif i<n-1 and tokens[i+1].startswith("##"):
            ind_sim_dict[i]=0
        else:
            ind_sim_dict[i] = find_sim(qc, token)
    items = ind_sim_dict.items()
    sorted_ind_sim_dict = sorted(items, key=lambda item: item[1], reverse=True)
    sorted_ind_sim_dict= [tup for tup in sorted_ind_sim_dict if tup[1] != 1]
    
    return sorted_ind_sim_dict

def replace_one_word_by_qc(qc_tokens, doc_tokens, tok_sim_dict, k): # k <= 20
    index = 0
    new_doc_tokens = doc_tokens.copy()
    print(f"*********new_doc_tokens = {new_doc_tokens}")
    print(f"*********qc_tokens = {qc_tokens}, index = {index}")
    new_doc_tokens[index:index] = qc_tokens
    return new_doc_tokens

def attack_qc_sim_query(args, qid, query_center, attack_pass_qid, orig_pass_qid, query, tokenizer, model_ref, model_name):
    to_save_query = []
    for j in range(len(attack_pass_qid)):
        print(f"****** attack_pass_qid_columns = {attack_pass_qid.columns}")
        pid = attack_pass_qid['passid'][j]
        print(f"***Attacking pid = {pid}")
        
        attack_pass = orig_pass_qid[orig_pass_qid['passid'] == pid]['text'].iloc[0]
        
        orig_score = get_score_model(query, attack_pass, model_ref, model_name)
        max_score = orig_score
        
        attack_pass_tokens = tokenizer.tokenize(attack_pass)
        orig_pass_len = len(attack_pass_tokens)
        qc_tokens = tokenizer.tokenize(query_center)
        qc_sim_indices = 0
        # attacked_pass = attack_pass
        # output format = qid\tpassid\torig_pass_len\tattack_word_number\tnew_score\t[DIV]\ttop_k_word\t[DIV]\tsubword_to_word\t[DIV]\tword_to_word*_list(sep by ', ')\t[DIV]\tattack_pass
        to_write = f"{qid}\t{pid}\t{orig_pass_len}"
        
        k = 0
        attack_num = 0
        max_attack_word_number= args.max_attack_word_number
        while(k<max_attack_word_number):
            new_pass_tokens = replace_one_word_by_qc(qc_tokens, attack_pass_tokens, qc_sim_indices, k)
            new_pass = tokenizer.convert_tokens_to_string(new_pass_tokens)
            new_score = get_score_model(query, new_pass, model_ref, model_name)
            if new_score > max_score:
                max_score = new_score
                attack_pass = new_pass
                attack_pass_tokens = new_pass_tokens
                attack_num += 1
            k += 1
        
        to_write = f"{qid}\t{pid}\t{orig_pass_len}\t{attack_num}\t{max_score}\t"+attack_pass
        to_save_query.append(to_write)
    return to_save_query
        


def attack_qc_sim(args, query_file, orig_pass, attack_qp_pass, tokenizer, model_ref, model_name):
    
    for i in range(len(query_file)):
        qid = query_file['qid'][i]
        query = query_file['query'][i]
        orig_pass_qid = orig_pass[orig_pass['qid']==qid].reset_index(drop=True)
        attack_pass_qid = attack_qp_pass[attack_qp_pass['qid']==qid].reset_index(drop=True)
        query_center, simm = get_query_center(query, args.embed_path, tokenizer)
        print(f"*** query = {query}")
        print(f"*** query_center = {query_center}")
        
        to_save_query_list = attack_qc_sim_query(args, qid, query_center, attack_pass_qid, orig_pass_qid, query, tokenizer, model_ref, model_name)
        with open(args.save_pass_tokens_path,'a') as file:
            for line in to_save_query_list:
                file.write(line + '\n')
        
        
        





#========================================================================

def run_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--msmarco_dir", type=str,
                        default=f"./data/msmarco-passage")
    
    parser.add_argument("--no_cuda", action='store_true')
    
    parser.add_argument("--previous_done", type=int, default=0)
    
    parser.add_argument("--max_attack_word_number", type=int, default=1)

    parser.add_argument("--model_name",type=str,
                        default='Luyu')
    
    parser.add_argument("--attack_dataset", type=str, default="trec_dl_2019")

    parser.add_argument("--save_file_name", type=str,default="sim_ow_02") 
    
    
    
    parser.add_argument("--embed_path", type=str,
                        default='./models/counter-fitted-vectors.txt')
    
    parser.add_argument("--orig_pass_file", type=str, default="./reranked_files/ori_pass_files/trec_dl_2020")
    
    #------------------------------------------------------------------------
    # python3 start_st_bb_attack.py --model_name monoT5 --attack_dataset trec_dl_2020 --save_file_name start_ow_01
    
    #---------------------------------------------------------------------------
    
    args = parser.parse_args()

    #------------------------------------------------------------------------
    args.orig_pass_file = "./reranked_files/ori_pass_files/"
    if args.model_name == 'Luyu':
        args.orig_pass_file += f"{args.attack_dataset}"
    else:
        args.orig_pass_file += f"{args.model_name}_{args.attack_dataset}"
    args.query_file = f"{args.msmarco_dir}/{args.attack_dataset.replace('_','-')}/queries.tsv"
    args.model_path = f"./models/{args.model_name}"
    args.save_pass_tokens_path = f"./Attack_save/{args.model_name}/attacked_{args.attack_dataset}/one_word/{args.save_file_name}"
    args.model_ranked_list_score_path = f"./reranked_files/{args.model_name}_{args.attack_dataset}"
    args.attack_qp_path = f"./Attack_pass/{args.model_name}_{args.attack_dataset}"
    args.device = "given later"
    
    return args    
    

def main():
    args = run_parse_args()
    model_name = args.model_name

    # get model ref
    if model_name == 'Luyu':
        model_ref = RerankerForInference.from_pretrained(args.model_path) 
    elif model_name == 'monoT5':
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64' 
        if not pt.started():
            pt.init()
        model_ref = MonoT5ReRanker(model=args.model_path)
    elif model_name == 'Distilbert':
        model_ref = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')     
    
    
    attack_save_path = args.save_pass_tokens_path
    if not os.path.exists(attack_save_path):
        os.mknod(attack_save_path)
    check = 0
    with open(attack_save_path, 'r') as f:
        line = f.readline()
        print(f"***********line = {line}*")
        if line == '':
            print("true")
            check = 1
    if check == 1:
        with open(attack_save_path, 'a') as f:
            f.write("qid\tpassid\torig_pass_len\tattack_num\tscore\t[DIV]\ttopk_grad\t[DIV]\tsubword->original\t[DIV]\tword->word*:score_gain\t[DIV]\tpass*\n")
            
            
    # word_re = BERTWordRecover(embed_name=None, bert_tokenizer=bert_tokenizer, bert_vocab_path="hi", max_query_length=50, max_pass_length=100)
    
    orig_pass = pd.read_csv(args.orig_pass_file, sep='\t')
    print(orig_pass.columns)
    attack_qp_pass = pd.read_csv(args.attack_qp_path, sep='\t')
    query_file_col = ['qid', 'query']
    query_file = pd.read_csv(args.query_file, sep='\t',header=None, names=query_file_col)
    
    

    st = time.time()
    
    attack_qc_sim(args, query_file, orig_pass, attack_qp_pass, bert_tokenizer, model_ref, model_name)
    et = time.time()
    print(f"Total time taken = {(et - st)/60.0} minutes.")     
                    
            
    
    
    return 

print(__name__)
if __name__=='__main__':
    main()