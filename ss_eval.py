import transformers
import pandas as pd
transformers.logging.set_verbosity(transformers.logging.ERROR)
import tensorflow as tf
import tensorflow_hub as hub




# texts = ["Robert is very knowledgeable and engaging presenter who conveys clear messages to his audience. I had the pleasure to work with Robert and develop and deliver bespoke alternative asset management training to Senior Financial Advisors (IFAs) in Europe."] # 
# adv = ["Robert is very knowledgeable and engaging presenter who conveys clear messages to his audience. I had the pleasure to work with Robert and develop and deliver bespoke alternative asset management training to Senior Financial Advisors (IFAs) in Europe."] # 
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# clean_embeddings = embed(texts)

# adv_embeddings = embed(adv)
# cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))
# print(f"Similarity = {cosine_sim}")




def compute_ss(orig_pass_path, perturbed_pass_path):
    orig_text = []
    p_text = []
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    orig_pass = pd.read_csv(orig_pass_path ,sep='\t')
    print(orig_pass.columns)
    purturbed_pass = open(perturbed_pass_path, 'r') #
    
    # purturbed_pass_file_format  => qid passid  orig_pass_len  attack_num  score    [DIV]   topk_grad   [DIV]   word->word* [DIV]   pass*
    
    purturbed_pass.readline() # skip the header
    for line in purturbed_pass:
        l = line.split('\t')
        qid = int(l[0])
        pid = int(l[1])
        # print(pid)
        p_pass = str(l[-1])
        id = orig_pass.index[orig_pass['passid'] == pid].to_list()[0]
        o_pass = orig_pass['text'][id]
        orig_text.append(o_pass)
        p_text.append(p_pass)
    
    clean_embeddings = embed(orig_text)   
    adv_embeddings = embed(p_text)
    cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))
    print(f"Similarity = {cosine_sim}")
    return cosine_sim
    
    
data = "msmarco_dev_200"
model_name = 'Luyu'
file_name = 'sim_ow_01'

# cos_sim = compute_ss(f'./reranked_files/ori_pass_files/trec_dl_20{data}', f'./Attack_save/{model_name}/attacked_trec_dl_20{data}/one_word/{file_name}')
cos_sim = compute_ss(f'./reranked_files/ori_pass_files/{data}', f'./Attack_save/{model_name}/attacked_{data}/one_word/{file_name}')

with open(f"./Evaluation/{model_name}/sim_{data}",'a') as f:
    f.write(f"Average Similarity = {cos_sim}\n")  # write the similarity to
    
print(f"{model_name} : {data} : {file_name} : similarity = {cos_sim}")