import numpy as np
import os
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import ast
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = genai.GenerativeModel('gemini-1.5-flash')


def load_merged_dfs(merged_folder):
    merged_dfs = {}
    for filename in os.listdir(merged_folder):
        merged_dfs[filename] = pd.read_csv(merged_folder+filename)
    return merged_dfs


def get_embeddings(extraction, annot_list, model_type):
    if model_type=='google':
        emb_extr = np.array(genai.embed_content('models/text-embedding-004', extraction)['embedding']).reshape(1, -1)
        emb_annot_list = [np.array(genai.embed_content('models/text-embedding-004', a)['embedding']).reshape(1, -1) for a in annot_list]
    elif model_type=='stella':
        model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", token=os.getenv("HUGGINGFACE_TOKEN"))
        emb_extr = np.array(model.encode(extraction)).reshape(1, -1)
        emb_annot_list = [np.array(model.encode(a)).reshape(1, -1) for a in annot_list]
    return emb_extr, emb_annot_list


def get_first_author(authors):
    return authors.split(' and')[0].lower()


def get_ids_citations(cit_df):
    authors = [get_first_author(auths) for auths in cit_df['Authors']]
    years = cit_df['Year']
    ids = ['{}_{}'.format(a, y) for (a, y) in zip(authors, years)]
    return ids


def get_ids_fp(fp_df):
    authors = [get_first_author(auths) for auths in fp_df['author']]
    years = fp_df['year']
    ids = ['{}_{}'.format(a, y) for (a, y) in zip(authors, years)]
    return ids


def get_ids_rct(rct_df):
    authors = rct_df['author name']
    years = rct_df['year']
    ids = ['{}_{}'.format(a, y) for (a, y) in zip(authors, years)]
    return ids


def get_ids(df, df_type):
    if df_type == 'citations':
        return get_ids_citations(df)
    elif df_type == 'fp':
        return get_ids_fp(df)
    elif df_type == 'rct':
        return get_ids_rct(df)


def get_lev(titles1, titles2):
    dists = np.zeros((len(titles1), len(titles2)))
    for i, t1 in enumerate(titles1):
        for j, t2 in enumerate(titles2):
            dists[i,j] = nltk.edit_distance(t1, t2)
    return dists


def get_matches(titles1, titles2, threshold=20):
    dists = get_lev(titles1, titles2)
    return np.sum(dists<threshold)