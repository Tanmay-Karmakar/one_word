import pandas as pd  # type: ignore
from numpy import mean
import argparse



def run_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_ranked_list_path", type=str, default= "./reranked_files/monoBert_trec_dl_2019")
    # parser.add_argument("--qid", type=int, default= 57)
    # parser.add_argument("--no_of_pairs", type=int, default= 1000)
    parser.add_argument("--interval_len", type=int, default=10)
    parser.add_argument("--starting_pos", type=int, default= 5)
    parser.add_argument("--output_dir", type=str, default= './Attack_pairs/interval_pairs_2019')
    

    args = parser.parse_args()
    return args

def main():
    args = run_parse_args()
    out = open(args.output_dir , 'w')
    inp = open(args.model_ranked_list_path, 'r')
    l = inp.readline()
    out.write(l)
    for i in range(args.starting_pos-1):
        inp.readline()

    i = 0
    for line in inp:
        if i % args.interval_len == 0:
            out.write(line)
        i+=1
    
if __name__ == '__main__':
    main()
    