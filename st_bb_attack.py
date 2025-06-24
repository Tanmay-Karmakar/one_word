import copy
import logging, argparse, os, time
from tqdm import tqdm
import torch 
from torch.utils.data import DataLoader
from transformers import (BertConfig, BertTokenizer)
from transformers import BertTokenizer
from reranker import RerankerForInference
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
import time
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import pandas as pd
from get_query_center import get_query_center


from myutils.semantic_helper import SemanticHelper
from utils import read_qps
from modeling import RankingBERT_Train
from marcopassage.dataset import MSMARCODataset_white_attack, get_collate_function, CollectionDataset
from myutils.word_recover.Bert_word_recover import BERTWordRecover
from myutils.attacker.attacker import Attacker


stop_words = set(stopwords.words('english'))
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)

def read_resources(embed_path, embed_cos_matrix_path, embed_name,
                   attack_qp_path, model_ranked_list_score_path, bert_tokenizer,
                   bert_vocab_path, max_query_length, max_pass_length, collection_memmap_dir):
    
    synonym_helper = SemanticHelper(embed_path, embed_cos_matrix_path)
    synonym_helper.build_vocab()
    synonym_helper.load_embedding_cos_sim_matrix()

    word_re = BERTWordRecover(embed_name, bert_tokenizer, bert_vocab_path, max_query_length, max_pass_length)

    attack_qps = read_qps(attack_qp_path)

    ori_ranked_list_qps = read_qps(model_ranked_list_score_path)

    collection = CollectionDataset(collection_memmap_dir)

    return synonym_helper, word_re, attack_qps, ori_ranked_list_qps, collection 


def remove_input_tensor_column(input_tensor, delete_index):

    res_tensor = input_tensor[
        torch.arange(input_tensor.size(0)) != delete_index]
    return res_tensor

def find_attack_pass_input_and_remove(batch_list, passid_list, attack_pass_id):

    res_batch_list = copy.deepcopy(batch_list)
    attack_pass_input = {}
    assert len(res_batch_list) == len(passid_list)
    for i in range(len(passid_list)):
        
        passids = passid_list[i]
        for j in range(len(passids)):
            pid = passids[j]
            if pid == attack_pass_id:
                attack_pass_input["input_ids"] = res_batch_list[i]['input_ids'][
                    j]
                attack_pass_input["token_type_ids"] = \
                res_batch_list[i]['token_type_ids'][j]
                attack_pass_input["position_ids"] = \
                res_batch_list[i]['position_ids'][j]

                res_batch_list[i]['input_ids'] = remove_input_tensor_column(
                    res_batch_list[i]['input_ids'], j)
                res_batch_list[i][
                    'token_type_ids'] = remove_input_tensor_column(
                    res_batch_list[i]['token_type_ids'], j)
                res_batch_list[i]['position_ids'] = remove_input_tensor_column(
                    res_batch_list[i]['position_ids'], j)

    return attack_pass_input, res_batch_list


def attack_by_testing_model(model, batch_list, passid_list,
                               attack_pass_id, args, attack_word_number,
                               bert_tokenizer, semantic_helper, word_re,
                               ori_score, ori_qps, qid, model_ref,model_name, query_center_tok_id):
    """
    Attacks the model by testing it on a batch of passages and finding the most vulnerable words to perturb.

    Args:
        model (nn.Module): The model to attack.
        batch_list (list): A list of batches of passages.
        passid_list (list): A list of passage IDs corresponding to the batches.
        attack_pass_id (int): The ID of the passage to attack.
        args (argparse.Namespace): The arguments for the attack.
        attack_word_number (int): The number of words to attack.
        bert_tokenizer (BertTokenizer): The BERT tokenizer.
        semantic_helper (SemanticHelper): The semantic helper.
        word_re (BERTWordRecover): The word recover.
        ori_score (float): The original score of the passage.
        ori_qps (dict): The original query passage scores.
        qid (int): The query ID.
        model_ref (RerankerForInference or MonoT5ReRanker): The model reference.
        model_name (str): The name of the model.

    Returns:
        tuple: A tuple containing the new passage token ID list, the score, the rank, the number of attacked words, the original top-k words, the attacked word list, and the original passage length.
    """
    attack_pass_input, batch_list = find_attack_pass_input_and_remove(batch_list,
                                                                    passid_list,
                                                                    attack_pass_id)

    attack_input_ids_list = attack_pass_input['input_ids'].tolist()
    sep_token_id = bert_tokenizer.sep_token_id

    query_token_id = attack_input_ids_list[1:attack_input_ids_list.index(sep_token_id)] #query

    with_last_sep_pass_token_ids_list = attack_input_ids_list[attack_input_ids_list.index(
        sep_token_id) + 1:]
    ori_pass_token_ids_list = with_last_sep_pass_token_ids_list[:len(with_last_sep_pass_token_ids_list) - 1]

    pass_token_ids_list = list({}.fromkeys(ori_pass_token_ids_list).keys())

    word_embedding_matrix = word_re.get_word_embedding(model)
    ori_we_matrix = word_embedding_matrix.clone().detach()


    attacker = Attacker()

    attacker.get_model_gradient(model, batch_list, attack_pass_input, args.device)
    if args.imp_k == 'bottom':
        gradient_norm_topk_word, gradient_topk_word_idx_list = word_re.get_lowest_gradient_words(
            model, attack_word_number, pass_token_ids_list)
    else:
        gradient_norm_topk_word, gradient_topk_word_idx_list = word_re.get_highest_gradient_words(
            model, attack_word_number, pass_token_ids_list)

    gradient_topk_words = [word_re.idx2word[word_idx] for word_idx in gradient_topk_word_idx_list]   # words which are to be replaced
    
    #-----------
    gradient_topk_words2 = []
    for i in range(len(gradient_topk_words)-1,-1,-1):
        word = gradient_topk_words[i]
        if word in stop_words:
            gradient_topk_word_idx_list.pop(i)

        else:
            gradient_topk_words2.append(word)

    

    #---------------

    # print(f"********highest- grad_ word = {gradient_topk_words2}")

    orig_topk_words = ", ".join(list(gradient_topk_words2))

    attacker.attack(model, batch_list, attack_pass_input,
                    attack_word_idx=pass_token_ids_list,
                    args=args, eps=args.eps, max_iter=args.max_iter)

    attacked_we_matrix = word_re.get_word_embedding(model)

    sim_word_ids_dict, sim_values, sub_word_dict,sub_to_orig = semantic_helper.pick_most_similar_words_batch(
        gradient_topk_words2, ori_pass_token_ids_list, word_re,
        args.simi_candi_topk, args.simi_threshod)
    if args.one_word and args.boo:
        new_pass_token_id_list, score, rank , atk_word_num, attack_word_list, orig_pass_len= word_re.recover_document_greedy_rank_pos_boo(
        ori_pass_token_ids_list, ori_we_matrix, attacked_we_matrix, sim_word_ids_dict,
        ori_score, model_name, query_token_id, args, sub_word_dict, ori_qps, attack_pass_id, qid,model_ref, query_center_tok_id)
    else:
        new_pass_token_id_list, score, rank , atk_word_num, attack_word_list, orig_pass_len= word_re.recover_document_greedy_rank_pos(
            ori_pass_token_ids_list, ori_we_matrix, attacked_we_matrix, sim_word_ids_dict,
            ori_score, model_name, query_token_id, args, sub_word_dict, ori_qps, attack_pass_id, qid,model_ref, query_center_tok_id)

    return new_pass_token_id_list, score, rank, atk_word_num, orig_topk_words , attack_word_list , orig_pass_len, sub_to_orig


def run_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=f"./data/train")

    parser.add_argument("--msmarco_dir", type=str,
                        default=f"./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str,
                        default="./data/collection_memmap")

    parser.add_argument("--max_query_length", type=int, default=32)

    parser.add_argument("--max_pass_length", type=int, default=256)

    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--data_num_workers", default=0, type=int)

    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--bert_tokenizer_path",
                        default='bert-base-uncased',
                        type=str)
    
    parser.add_argument("--pass_list_length", type=int,
                        default=100)


    parser.add_argument("--eps", type=float, default=45)

    parser.add_argument("--max_iter", type=int, default=3)

    parser.add_argument("--simi_candi_topk", type=int, default=50)

    parser.add_argument("--simi_threshod", type=float, default=0.7)

    parser.add_argument("--previous_done", type=int, default=0)

    parser.add_argument("--max_attack_word_number", type=int, default=20)

    parser.add_argument("--find_word_search", action="store_true")

    parser.add_argument("--embed_path", type=str,
                        default='./models/counter-fitted-vectors.txt')

    parser.add_argument("--embed_cos_matrix_path", type=str,
                        default='./models/sim-emb-mat.npy')
    
    parser.add_argument("--batch_size", default=25, type=int)
    
    parser.add_argument("--qc", type= str, default="query_center")

#----------

    parser.add_argument("--imp_k", type=str, default='top')
    
    parser.add_argument("--boo", action= 'store_true')

    parser.add_argument("--model_name",type=str,
                        default='Luyu')
    
    parser.add_argument("--attack_dataset", type=str, default="msmarco_dev_200")

    parser.add_argument("--save_file_name", type=str,default="boo_01") 
    
    parser.add_argument("--one_word", action='store_true')
    
    parser.add_argument("--one_word_type", type=str, default='grad') # grad or sim or start
    

# python3 st_bb_attack.py --attack_dataset trec_dl_2020 --model_name monoT5 --save_file_name top_prada_2020 
# ./Attack_save/monoT5/attacked_trec_dl_2019/ er moddhe one_word naame ekta folder banate hbe
#-----------
    # parser.add_argument("--model_path", type=str,
    #                     default='./models/Luyu')

    # parser.add_argument("--save_pass_tokens_path", type=str,
    #                     default='./Attack_save/Luyu/attacked_trec_dl_2020/low_test') 
    
    # parser.add_argument("--model_ranked_list_score_path", type=str,
    #                     default='./reranked_files/rem') # ranked list(tsv file) format => qid\tpassid\tscore\trank [there should be 100 passages for each query and sorted wrt rank. If some query has less passages attack on those passages have to be done seperately]
    
    # parser.add_argument("--attack_qp_path", type=str,
    #                     default="./Attack_pass/test_attack") # format(tsv file) => qid\tpassid\tscore\trank
    
    # parser.add_argument("--tokenize_dir", type=str, default="./new_tokenize/luyu_token/trec-dl-2020-queries.tokenized.json")

    args = parser.parse_args()
    # python3 st_bb_attack.py --imp_k bottom --model_name Distilbert --attack_dataset trec_dl_2020 --save_file_name low_test
    args.one_word_replace_type = args.one_word_type
    args.one_word_replace = args.one_word
    args.query_file = f"{args.msmarco_dir}/{args.attack_dataset.replace('_','-')}/queries.tsv"
    args.model_path = f"./models/{args.model_name}"
    if args.one_word:
        args.save_pass_tokens_path = f"./Attack_save/{args.model_name}/attacked_{args.attack_dataset}/one_word/{args.save_file_name}"
    else:
        args.save_pass_tokens_path = f"./Attack_save/{args.model_name}/attacked_{args.attack_dataset}/{args.save_file_name}"
    args.model_ranked_list_score_path = f"./reranked_files/{args.model_name}_{args.attack_dataset}"
    args.attack_qp_path = f"./Attack_pass/{args.model_name}_{args.attack_dataset}"
    # args.attack_qp_path = f"./Attack_pass/test_attack"
    args.tokenize_dir = f"./new_tokenize/{args.model_name}_token/{args.attack_dataset}_queries.tokenized.json"






    # python3 st_bb_attack.py --model_name monoT5 --model_path ./models/monoT5 --save_pass_tokens_path ./Attack_save/monoT5/attacked_trec_dl_2020/test --model_ranked_list_score_path ./reranked_files/monoT5_trec_dl_2020 --attack_qp_path ./Attack_pass/monoT5_trec_dl_2020 --tokenize_dir new_tokenize/T5token/trec-dl-2020-queries.tokenized.json --previous_done 47

    time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
    args.log_dir = f"{args.output_dir}/log/{time_stamp}"
    args.model_save_dir = f"{args.output_dir}/models"
    args.bert_vocab_path = './' + args.bert_tokenizer_path + '_/vocab.txt' 
    # args.bert_vocab_path = './models/monoT5/vocab.txt'
    assert args.pass_list_length % args.batch_size == 0
    return args


def main():
    start_time = time.time()
    args = run_parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    pass_list_length = args.pass_list_length 
    assert pass_list_length % args.batch_size == 0 
    read_num = int(pass_list_length / args.batch_size)  
    # load model
    model_path = args.model_path 

    config = BertConfig.from_pretrained(model_path)
    model = RankingBERT_Train.from_pretrained(model_path, config=config)
    model.to(args.device)
    # multi-gpu
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # restore the model's state
    model_state_dict = model.state_dict()
    model_state_dict = copy.deepcopy(model_state_dict)

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

    
    # get the embedding layer's name
    for name, param in model.named_parameters():
        args.embed_name = name
        break
    print(f"embed_name = {args.embed_name}")

    logger.info("evaluation parameters %s", args)

    # create global resources
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)  
    synonym_helper, word_recover, attack_qps, ori_qps, collection = read_resources \
        (args.embed_path, args.embed_cos_matrix_path, args.embed_name,
         args.attack_qp_path, args.model_ranked_list_score_path, bert_tokenizer, args.bert_vocab_path,
         args.max_query_length, args.max_pass_length, args.collection_memmap_dir)

    max_attack_word_number = args.max_attack_word_number
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

    attacked_pass_dict = {}
    attacked_pass_score_dict = {}
    attacked_pass_rank_dict ={}
    attacked_pass_num_dict ={}
    attacked_pass_orig_topk_dict ={}
    attacked_word_list_dict ={}
    orig_pass_length_dict ={}
    subword_to_original = {}


    # create dataset
    mode = 'dev'
    dev_dataset = MSMARCODataset_white_attack(mode, args.model_ranked_list_score_path,
                                 args.collection_memmap_dir, args.tokenize_dir,
                                 args.bert_tokenizer_path,
                                 args.max_query_length, args.max_pass_length)
    collate_fn = get_collate_function(mode=mode)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                num_workers=args.data_num_workers,
                                collate_fn=collate_fn)
    dataloader_iter = enumerate(dev_dataloader)

    save_attacked_pass_f = open(attack_save_path, 'a')
    previous_done = args.previous_done

    # skip some queries 
    for j in range(previous_done * read_num):
        dataloader_iter.__next__()
    
    qip_list_t = tqdm(list(attack_qps.keys())[previous_done:])

    tested_q_num = 0
    col = ['qid', 'query']
    query_file = pd.read_csv(args.query_file, sep='\t', header=None, names=col)
    tokeni_zer = BertTokenizer.from_pretrained("bert-base-uncased")
    for qid in qip_list_t:
        print(f"@@@@@@@@@@@@@@@@ QID = {qid} *")
        
        tested_q_num += 1

        attack_passid_list = list(attack_qps[qid].keys())
        batch_list = []
        passid_list = []
        for i in range(read_num):
            batch_index, data = dataloader_iter.__next__()
            batch, qids, passids = data
            batch_list.append(batch)
            passid_list.append(passids)

        attack_passid_list_t = tqdm(attack_passid_list)
        query_center_token_id = None
        if args.one_word_replace:
            # attack_pass_input, batch_list = find_attack_pass_input_and_remove(batch_list,
            #                                                         passid_list,
            #                                                         attack_pass_id)

            # attack_input_ids_list = attack_pass_input['input_ids'].tolist()
            # sep_token_id = bert_tokenizer.sep_token_id

            # query_token_id = attack_input_ids_list[1:attack_input_ids_list.index(sep_token_id)] #query
            query_text = query_file[query_file['qid']==qid]['query'].iloc[0]
            print(f"query_text = {query_text}")
            query_center, simm = get_query_center(query_text, args.embed_path, tokeni_zer)
            print(f"query_center = {query_center}, sim = {simm}")
            args.qc = query_center
            query_center_token = tokeni_zer.tokenize(query_center)
            query_center_token_id = tokeni_zer.convert_tokens_to_ids(query_center_token)
            
            

        for attack_passid in attack_passid_list_t:
            print(f"@@@@@@@@@@@@@@@@@ attack_passid = {attack_passid}")

            model.load_state_dict(model_state_dict)
            ori_score = attack_qps[qid][attack_passid]
            new_pass_token_id_list, score, rank , atk_word_num, orig_topk_words ,attack_word_list , orig_pass_len, sub_to_orig= attack_by_testing_model(model,
                                                                      batch_list,
                                                                      passid_list,
                                                                      attack_passid, args,
                                                                      max_attack_word_number,
                                                                      bert_tokenizer,
                                                                      synonym_helper,
                                                                      word_recover,
                                                                      ori_score,
                                                                      ori_qps, qid,model_ref, model_name, query_center_token_id)
            attack_pass_key = str(qid) + '_' + str(attack_passid)
            attacked_pass_dict[attack_pass_key] = new_pass_token_id_list
            attacked_pass_score_dict[attack_pass_key] = score
            attacked_pass_rank_dict[attack_pass_key] = rank
            attacked_pass_num_dict[attack_pass_key] = atk_word_num
            attacked_pass_orig_topk_dict[attack_pass_key] = orig_topk_words
            attacked_word_list_dict[attack_pass_key] = attack_word_list
            orig_pass_length_dict[attack_pass_key] = orig_pass_len
            subword_to_original[attack_pass_key] = sub_to_orig

        for qid_passid in attacked_pass_dict:
            attacked_pass = word_recover.recover_doc(qid_passid.split('_')[1], attacked_pass_dict[qid_passid],
                                                    collection, args.max_pass_length)
            # output format = qid\tpassid\torig_pass_len\tattack_word_number\tnew_score\t[DIV]\ttop_k_word\t[DIV]\tsubword_to_word\t[DIV]\tword_to_word*_list(sep by ', ')\t[DIV]\tattack_pass
            to_write = qid_passid.split('_')[0] + '\t' + qid_passid.split('_')[1] + '\t' + str(orig_pass_length_dict[qid_passid]) + '\t' + str(attacked_pass_num_dict[qid_passid]) + '\t' + str(attacked_pass_score_dict[qid_passid]) + '\t' + '[DIV]' + '\t' + str(attacked_pass_orig_topk_dict[qid_passid]) + '\t' + '[DIV]' + '\t' + ", ".join(subword_to_original[qid_passid]) + '\t' + '[DIV]' + '\t' +  ", ".join(attacked_word_list_dict[qid_passid]) + '\t' + '[DIV]' + '\t' + attacked_pass 
                      
            save_attacked_pass_f.write(to_write + '\n')
        attacked_pass_dict = {}

    
    end_time = time.time()
    print(f'Total time taken = {end_time - start_time} seconds.')
    
if __name__ == "__main__":
    main()
