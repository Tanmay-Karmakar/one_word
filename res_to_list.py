import pandas as pd

def res_to_rank_list(res_path,list_path):
    df = pd.read_csv(res_path, sep = '\t', header = None, names = ['qid', 'Q0', 'passid', 'rank', 'score', 'distilbert'])
    rank_list_1000 = df.iloc[:,[0,2,4,3]]
    rank_list_100 = rank_list_1000.groupby('qid', group_keys=False).apply(lambda x: x.head(100)).reset_index()
    rank_list_100 = rank_list_100.drop(axis = 'columns',labels = ['index'])
    rank_list_100.to_csv(list_path, sep='\t', index = False)
    print(rank_list_100)
    




res_to_rank_list("./res_files_trec-dl-19-20/distilbert_res_2019.res", './reranked_files/Distilbert_trec_dl_2019')
