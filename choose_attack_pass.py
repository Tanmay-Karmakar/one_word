import pandas as pd
import argparse

def run_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_ranked_list_path", type=str, default= "./reranked_files/Distilbert_trec_dl_2020")
    parser.add_argument("--starting_pos", type=int, default= 11)
    parser.add_argument("--output_dir", type=str, default= './Attack_pass/Distilbert_trec_dl_2020')
    

    args = parser.parse_args()
    return args

def choose_pass(starting_pos, rerank_file, output_dir):
    rerank_df = pd.read_csv(rerank_file , sep='\t')
    unique_qids = rerank_df['qid'].unique()
    for qid in unique_qids:
        df_for_q = rerank_df.loc[rerank_df['qid'] == qid].reset_index(drop=True)
        df_for_q = df_for_q.iloc[starting_pos-1:]
        df_for_q.to_csv(output_dir, mode = 'a' ,sep='\t' , index = False,header = False)

def main():
    args = run_parse_args()
    with open(args.output_dir, 'w') as f:
        f.write("qid\tpassid\tscore\trank\n")

    choose_pass(args.starting_pos, args.model_ranked_list_path, args.output_dir)

    

if __name__ == '__main__':
    main()
