import ir_datasets
import pandas as pd



rerank_file_path = "./reranked_files/monoT5_trec_dl_2019"
output_path = "./reranked_files/ori_pass_files/monoT5_trec_dl_2019"
out = open(output_path, 'a')
out.write('qid\tpassid\ttext\n')
dataset = ir_datasets.load("msmarco-passage")

rerank_file = pd.read_csv(rerank_file_path , sep='\t')
for i in rerank_file.index.values:
    qid = rerank_file['qid'][i]
    pid = rerank_file['passid'][i]
    print(f"pid = {pid}")
    for doc in dataset.docs_iter():
        # print(doc.doc_id)
        if doc.doc_id == str(pid):
            text = doc.text
            print(f"done {i}; passid {pid}")
            break
    out.write(f"{qid}\t{pid}\t{text}\n")

out.close()