from sentence_transformers import SentenceTransformer
from pyterrier_t5 import MonoT5ReRanker
from reranker import RerankerForInference
import pandas as pd
import os
import pyterrier as pt
from sentence_transformers import util

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


def get_model_ref(model_name):
    if model_name == 'Luyu':
        model_ref = RerankerForInference.from_pretrained("./models/Luyu") 
    elif model_name == 'monoT5':
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
        if not pt.started():
            pt.init()
        model_ref = MonoT5ReRanker(model='./models/monoT5')
    elif model_name == 'Distilbert':
        model_ref = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    return model_ref

# sentence-transformers/msmarco-distilbert-base-tas-b

def get_score(query, pas, model_name):
    model_ref = get_model_ref(model_name)
    if model_name == 'Luyu':
        return get_score_luyu(query, pas, model_ref)
    elif model_name == 'Distilbert':
        return get_score_distilbert(query, pas, model_ref)
    elif model_name == 'monoT5':
        return get_score_monoT5(query, pas, model_ref)


model_name = 'Distilbert'
query = "dog day afternoon meaning"
d1 = "The afternoon is the part of each day that begins at noon or lunchtime and ends at about six o'clock, or after it is dark in winter. 1. the present day. You refer to the afternoon of the present day as this afternoon. I rang Pat this afternoon."
d2 = "The afternoon are the parts of each days that begains at noons or lanchtime and ends at about six o'clock, of after it is dark in winters. 1. the presents day. You refers to the afternoon of the presents day as this afternoon. I rang Pat this afternoon"
score1 = get_score(query, d1, model_name)
score2 = get_score(query, d2, model_name)

print(f"score1 = {score1}; score2 = {score2}\nDiff in score = score2 - score1 = {score2 - score1}")