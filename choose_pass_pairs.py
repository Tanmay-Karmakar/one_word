import argparse, os
import pandas as pd



def run_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_ranked_list_path", type=str, default= "./reranked_files/luyu_trec_dl_2019")
    # parser.add_argument("--qid", type=int, default= 57)
    parser.add_argument("--no_of_pairs", type=int, default= 1000)
    parser.add_argument("--gap_between_pass", type=int, default=5)
    parser.add_argument("--starting_pos", type=int, default= 5)
    parser.add_argument("--output_dir", type=str, default= './Attack_pairs/trec_dl_2019.tsv')
    

    args = parser.parse_args()
    return args

def main():
    args = run_parse_args()
    rerank_df = pd.read_csv(args.model_ranked_list_path, sep= "\t")
    # total_pass = len(rerank_df['passid'])

    out = open(args.output_dir, 'w')
    out.write("qid\tpassid_1\tscore_1\trank_1\tpassid_2\tscore_2\trank_2\n")

    # qid = args.qid
    starting_pos = args.starting_pos
    gap = args.gap_between_pass
    no_of_pairs = args.no_of_pairs

    # starting_pos = rerank_df.index[(rerank_df['qid'] == qid)].to_list()[starting_pos-1]
    unique_qids = rerank_df['qid'].unique()
    # print(f"******unique_qids are = {unique_qids}, number of unique_qids = {len(unique_qids)}******")
    for qid in unique_qids:
        df_for_q = rerank_df.loc[rerank_df['qid'] == qid].reset_index(drop=True)
        total_pass = len(df_for_q['qid'])
        if starting_pos >= total_pass-gap:
            continue
        for i in range(starting_pos, min(total_pass - gap , starting_pos+ no_of_pairs)):
            out.write(f"{str(df_for_q['qid'][i])}\t{str(df_for_q['passid'][i])}\t{str(df_for_q['score'][i])}\t{str(df_for_q['rank'][i])}\t{str(df_for_q['passid'][i+gap])}\t{str(df_for_q['score'][i+gap])}\t{str(df_for_q['rank'][i+gap])}\n")
            
    out.close()
    print(f"Done with {args.output_dir}")

if __name__ == "__main__":
    main()