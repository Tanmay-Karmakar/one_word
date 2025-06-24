import pandas as pd  # type: ignore
import argparse


def calc_avg_eval_for_each_query(rerank_list_path, attack_pass_list_path, gap):
    '''
        returns 3 dicts of average eval for each query => {qid:avg_eval}
    '''
    avg_success_rates_list_dict = {}  # format-> {qid: avg_success rate}
    avg_score_boost_list_dict = {}  # fromat-> {qid: avg_score boost}
    avg_rank_boost_list_dict = {}   # format-> {qid: avg_rank boost}

    rerank_list = pd.read_csv(rerank_list_path, sep='\t')
    attack_pass_list = open(attack_pass_list_path, 'r')
    # attacked output format -> qid passid  orig_pass_len  attack_num  score    [DIV]   topk_grad   [DIV]   word->word* [DIV]   pass*
    attack_pass_list.readline() 


    for line in attack_pass_list:
        l = line.split('\t')
        qid = int(l[0])
        pid = int(l[1])

        new_score = float(l[4])
        reli = rerank_list.loc[rerank_list['qid'] == qid].reset_index()
        id = reli.index[reli['passid'] == pid].to_list()[0]
        pair_id = id - gap # index of D_j

        if pair_id < 0:
            continue
        pair_rank = reli['rank'][pair_id]
        pair_score = reli['score'][pair_id]
        r = 1
        for i in reli['score']:
            if i < new_score:           # Calculate the new rank of the perturbed passage
                break
            r += 1
        new_rank = r
        r = 1
        
        if qid not in avg_success_rates_list_dict:
            avg_success_rates_list_dict[qid] = []
            avg_score_boost_list_dict[qid] = []
            avg_rank_boost_list_dict[qid] = []


        if new_rank < pair_rank:
            avg_success_rates_list_dict[qid].append(1)
            avg_score_boost_list_dict[qid].append(new_score - pair_score)
            avg_rank_boost_list_dict[qid].append(pair_rank - new_rank)
    
        else:
            avg_success_rates_list_dict[qid].append(0)
            avg_score_boost_list_dict[qid].append(0)
            avg_rank_boost_list_dict[qid].append(0)
    attack_pass_list.close()

    avg_success_rates = {}
    avg_score_boost = {}
    avg_rank_boost = {}

    for q in avg_success_rates_list_dict:
        avg_success_rates[q] = sum(avg_success_rates_list_dict[q])/len(avg_success_rates_list_dict[q])

    for q in avg_score_boost_list_dict:
        avg_score_boost[q] = sum(avg_score_boost_list_dict[q])/len(avg_score_boost_list_dict[q])

    for q in avg_rank_boost_list_dict:
        avg_rank_boost[q] = sum(avg_rank_boost_list_dict[q])/len(avg_rank_boost_list_dict[q])

    return avg_success_rates , avg_score_boost, avg_rank_boost


        
def calc_overall_avg_eval(rerank_list_path, attack_pass_list_path, gap):
    sus, sco, rk = calc_avg_eval_for_each_query(rerank_list_path, attack_pass_list_path, gap)

    print(f"success rates = {sus}")
    l = len(sus)
    s = 0
    for i in sus:
        s = s+sus[i]
    avg_sus_rate = s/l
    print(f"Average success rate = {avg_sus_rate}")

    print(f"score boosts = {sco}")
    s = 0
    l = len(sco)
    for i in sco:
        s = s+sco[i]
    avg_sco_boost = s/l
    print(f"Average score boost = {avg_sco_boost}")

    print(f"rank boosts = {rk}")
    s = 0
    l = len(rk)
    for i in rk:
        s = s+rk[i]
    avg_rk_boost = s/l
    print(f"Average rank boost = {avg_rk_boost}")



    return avg_sus_rate, avg_sco_boost, avg_rk_boost


def calc_pp_for_each_query(attack_pass_list_path):
    '''
        returns a dict of average perturbation percentage for each query => {qid:avg-pp}
    '''
    pp_list_dict = {}
    attack_pass_list = open(attack_pass_list_path, 'r')
    attack_pass_list.readline()
    for line in attack_pass_list:
        l = line.split('\t')
        qid = int(l[0])
        pid = int(l[1])
        orig_pass_len = int(l[2])
        perturbed_word_num = int(l[3])
        pp_for_pass = (perturbed_word_num/orig_pass_len)*100
        if not qid in pp_list_dict:
            pp_list_dict[qid] = []
        pp_list_dict[qid].append(pp_for_pass)
    attack_pass_list.close()
    
    pp_dict = {}
    for qid in pp_list_dict:
        pp_dict[qid] = (sum(pp_list_dict[qid])/len(pp_list_dict[qid]) , len(pp_list_dict[qid]))

    return pp_dict

def calc_avg_pp(attack_pass_list_path):
    pp_dict = calc_pp_for_each_query(attack_pass_list_path)
    print(f"Perturbation Percentage = {pp_dict}")
    avg_pp_list = list(pp_dict.values())
    sum_pp_list = [i*j for (i,j) in avg_pp_list]
    sum_pp = sum(sum_pp_list)
    num_doc_list = [j for (_,j) in avg_pp_list]
    num_doc = sum(num_doc_list)

    avg_pp = sum_pp/num_doc

    #avg_pp = sum(pp_dict.values()[0])/len(pp_dict)
    print(f"Average Perturbation Percentage = {avg_pp}%")

    return avg_pp


def interval_avg_for_each_query(rerank_list_path, attack_pass_list_path, gap):
    avg_success_rates_dict_dict = {}    # {qid:{interval:[eval]}}
    avg_score_boost_dict_dict = {}
    avg_rank_boost_dict_dict = {}


    rerank_list = pd.read_csv(rerank_list_path, sep='\t')
    attack_pass_list = open(attack_pass_list_path, 'r')
    attack_pass_list.readline()  # The first line contains the headers qid passid  orig_pass_len  attack_num  score   topk_grad   word->word* pass*


    for line in attack_pass_list:
        l = line.split('\t')
        qid = int(l[0])
        pid = int(l[1])

        new_score = float(l[4])
        reli = rerank_list.loc[rerank_list['qid'] == qid].reset_index()
        id = reli.index[reli['passid'] == pid].to_list()[0]
        pair_id = id - gap
        if pair_id < 0:
            # pair_id = 0
            continue
        pair_rank = reli['rank'][pair_id]
        pair_score = reli['score'][pair_id]

        r = 1
        for i in reli['score']:
            if i < new_score:
                break
            r += 1
        new_rank = r

        if qid not in avg_rank_boost_dict_dict:
            avg_success_rates_dict_dict[qid] = {}
            avg_score_boost_dict_dict[qid] = {}
            avg_rank_boost_dict_dict[qid] = {}
        interval = (id)//10
        if interval not in avg_success_rates_dict_dict[qid]:
            avg_success_rates_dict_dict[qid][interval] = []
            avg_score_boost_dict_dict[qid][interval] = []
            avg_rank_boost_dict_dict[qid][interval] = []
        
        if new_rank < pair_rank:
            avg_success_rates_dict_dict[qid][interval].append(1)
            avg_score_boost_dict_dict[qid][interval].append(new_score - pair_score)
            avg_rank_boost_dict_dict[qid][interval].append(pair_rank - new_rank)
        
        else:
            avg_success_rates_dict_dict[qid][interval].append(0)
            avg_score_boost_dict_dict[qid][interval].append(0)
            avg_rank_boost_dict_dict[qid][interval].append(0)
    
    attack_pass_list.close()
   
    return avg_success_rates_dict_dict , avg_score_boost_dict_dict , avg_rank_boost_dict_dict

def avg_interval_eval(rerank_list_path, attack_pass_list_path, gap):
    int_success_rate, int_score_boost, int_rank_boost = interval_avg_for_each_query(rerank_list_path, attack_pass_list_path, gap)
    interval_success_rate = {}
    interval_score_boost = {}      # {interval:avg_for_interval}
    interval_rank_boost = {}        #{qid: {interval:[1,0,0,1,...]}}
    q = next(iter(int_success_rate))
    for i in int_success_rate[q]:
        sum_success_rate = 0
        len_success_rate = 0
        sum_score_boost = 0
        len_score_boost = 0
        sum_rank_boost = 0
        len_rank_boost = 0

        for qid in int_success_rate:
            if i not in int_success_rate[qid]:
                continue
            sum_success_rate += sum(int_success_rate[qid][i])
            len_success_rate += len(int_success_rate[qid][i])
            sum_score_boost += sum(int_score_boost[qid][i])
            len_score_boost += len(int_score_boost[qid][i])
            sum_rank_boost += sum(int_rank_boost[qid][i])
            len_rank_boost += len(int_rank_boost[qid][i])

        interval_success_rate[i] = sum_success_rate/len_success_rate
        interval_score_boost[i] = sum_score_boost/len_score_boost
        interval_rank_boost[i] = sum_rank_boost/len_rank_boost
    
    print(f"interval_success_rate = {interval_success_rate}")
    print(f"interval_score_boost = {interval_score_boost}")
    print(f"interval_rank_boost = {interval_rank_boost}")
    
    return interval_success_rate, interval_score_boost, interval_rank_boost


def avg_pp_pn_for_each_query(attack_pass_list_path):
    '''
        returns a dict of average perturbation percentage for each query => {qid:avg-pp}
    '''
    total_purturbed_word_num = 0
    total_doc_num = 0
    total_pp = 0
    attack_pass_list = open(attack_pass_list_path, 'r')
    attack_pass_list.readline()
    for line in attack_pass_list:
        total_doc_num += 1
        l = line.split('\t')
        qid = int(l[0])
        pid = int(l[1])
        orig_pass_len = int(l[2])
        perturbed_word_num = int(l[3])
        total_purturbed_word_num += perturbed_word_num
        pp_for_pass = (perturbed_word_num/orig_pass_len)*100
        total_pp += pp_for_pass

    avg_pp = total_pp/total_doc_num
    avg_pn = total_purturbed_word_num/total_doc_num
    print(f"avg_pp = {avg_pp}")
    print(f"avg_pn = {avg_pn}")
    return avg_pp, avg_pn

            

def avg_sr_rb_sb(args):
    success_rate,score_boost,rank_boost = calc_overall_avg_eval(rerank_list_path= args.model_ranked_list_path, attack_pass_list_path= args.attack_pass_path, gap= args.gap)
    out = open(args.output_dir, 'a')
    out.write(f"Gap Between Passages = {args.gap}\n")
    out.write(f"\t\t Average Pairwise Success Rate = {success_rate*100}\n")
    out.write(f"\t\t Average Pairwise Score Boost = {score_boost}\n")
    out.write(f"\t\t Average Pairwise Rank Boost = {rank_boost}\n\n")
    out.close()

def perturbed_percentage(args):
    # avg_pp = calc_avg_pp(attack_pass_list_path = args.attack_pass_path)
    avg_pp, avg_pn = avg_pp_pn_for_each_query(attack_pass_list_path = args.attack_pass_path)
    out = open(args.output_dir, 'a')
    out.write(f"Average Perturbation Percentage = {avg_pp}%\n")
    out.write(f"Average Perturbation Number = {avg_pn}\n")
    
    out.close()

def interval_eval(args):
    int_success_rate, int_score_boost, int_rank_boost = avg_interval_eval(rerank_list_path=args.model_ranked_list_path, attack_pass_list_path=args.attack_pass_path, gap=args.gap)
    out = open(args.output_dir, 'a')
    out.write(f"Gap Between Passages = {args.gap}\n")
    out.write(f"\tInterval Success_Rate:\n")
    for i in int_success_rate:
        out.write(f"\t\tInterval {(10*i +1 , 10*i + 10)} = {int_success_rate[i]*100}\n")
    
    out.write(f"\n\tInterval Score_Boost:\n")
    for i in int_score_boost:
        out.write(f"\t\tInterval {(10*i +1 , 10*i + 10)} = {int_score_boost[i]}\n")

    out.write(f"\n\tInterval Rank_Boost:\n")
    for i in int_rank_boost:
        out.write(f"\t\tInterval {(10*i +1 , 10*i + 10)} = {int_rank_boost[i]}\n")
    out.write('\n=======================================================================\n')
    out.close()




def run_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default= 'trec_dl_2019')
    parser.add_argument("--model", type=str, default= 'Luyu')
    parser.add_argument("--output_file", type=str, default= 'sim_ow_01')
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--what", type=str , default="all")
    parser.add_argument("--oo", action='store_true')

    # python3 eval.py --dataset trec_dl_2020 --model monoT5 --output_file top_prada_2020

    args = parser.parse_args()
    # parser.add_argument("--model_ranked_list_path", type=str, default= "./reranked_files/Luyu_trec_dl_2019")
    # parser.add_argument("--attack_pass_path", type=str , default='./Attack_save/Luyu/attacked_msmarco_dev_200/one_word_grad_01')
    # parser.add_argument("--output_dir", type=str, default= './Evaluation/Luyu/msmarco_dev_200_one_word_grad')
    
    args.model_ranked_list_path = f"./reranked_files/{args.model}_{args.dataset}"
    if args.oo:
        args.attack_pass_path = f"./Attack_save/{args.model}/attacked_{args.dataset}/one_word/{args.output_file}"
    else:
        args.attack_pass_path = f"./Attack_save/{args.model}/attacked_{args.dataset}/{args.output_file}"
    if args.oo:
        args.output_dir = f"./Evaluation/{args.model}/one_word/{args.dataset}_{args.output_file}"
    else:
        args.output_dir = f"./Evaluation/{args.model}/{args.dataset}_{args.output_file}"
    return args

def main():
    args = run_parse_args()
    if args.what == "avg-sr-rb-sb":
        avg_sr_rb_sb(args)
    elif args.what == "pp":
        perturbed_percentage(args)
        
    elif args.what == "interval-eval":
        interval_eval(args)
    elif args.what == "all":
        avg_sr_rb_sb(args)
        perturbed_percentage(args)
        interval_eval(args)

    else:
        raise KeyError("Oops!! Something went wrong. Use avg-sr-rb-sb, pp or interval-eval with '--what'")


if __name__=="__main__":
    
    main()

