import pandas as pd
import ast
import os
import argparse


def load_dfs(batch, file, extraction_folder='../data/extraction/'):
    dfs = {}
    dfs['citations'] = pd.read_csv(f'{extraction_folder}/parsed_citations/{batch}/{file}.csv')
    dfs['fp'] = pd.read_csv(f'{extraction_folder}/FP/{batch}/FP_{file}.csv').drop('Unnamed: 0', axis=1)
    dfs['rct'] = pd.read_csv(f'{extraction_folder}/RCT/{batch}/new_formatted_{file}.csv')
    for df_type, df in dfs.items():
        df['id'] = get_ids(df, df_type)
        df['id'] = df['id'].astype(str)
        dfs[df_type] = df
        print("Found {} entries in {}".format(len(df['id']), df_type))
    return dfs

def get_first_author(authors):
    return authors.split(',')[0]

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
    authors = [auth for auth in rct_df['author name']]
    years = rct_df['year']
    ids = ['{}_{}'.format(a, y) for (a, y) in zip(authors, years)]
    return ids

def get_ids_io(io_df):
    authors = [get_first_author(str(s_name)) for s_name in io_df['study name']]
    years = [re.findall('[0-9]{4}', str(s)) for s in io_df['study name']]
    years = [y[0] if len(y)>0 else None for y in years]
    ids = ['{}_{}'.format(a, y) for (a, y) in zip(authors, years)]
    return ids

def get_ids(df, df_type):
    if df_type == 'citations':
        return get_ids_citations(df)
    elif df_type == 'fp':
        return get_ids_fp(df)
    elif df_type == 'rct':
        return get_ids_rct(df)
    elif df_type == 'io_mentions' or df_type == 'io_tables':
        return get_ids_io(df)
    
def get_citations_em(dfs, left_type='fp'):
    merged_df = dfs[left_type].merge(dfs['citations'], on='id', how='left')
    proportion_found = merged_df[merged_df['Citation'].isna()].shape[0]/merged_df.shape[0]
    proportion_found = round(100*proportion_found,2)
    print("Found {}% of citations in forest plots via exact matching.".format(proportion_found))
    return merged_df


import re

def get_fp_cit_df(dfs):
    df1 = dfs['fp']
    df2 = dfs['citations']

    # Getting citation and title
    citations = []
    titles = []
    n_found = 0
    for i, row1 in df1.iterrows():
        authors = row1['author']
        if len(authors.split(' '))>1:
            author_list = re.findall('([A-Z][A-za-z]*)[^a-z]', authors)
        else:
            author_list = [authors]
        if len(author_list)>0:
            year = row1['year']
            found_citation = False
            for j, row2 in df2.iterrows():
                if not found_citation:
                    try:
                        if abs(int(year)-int(row2['Year']))<2:
                            found_authors = True
                            for author in author_list:
                                if not author in row2.Citation:
                                    found_authors = False
                            if found_authors:
                                found_citation = True
                                cur_citation = row2.Citation
                                cur_title = row2.Title
                    except Exception:
                        found_authors = True
                        for author in author_list:
                            if not author in row2.Citation:
                                found_authors = False
                        if found_authors:
                            found_citation = True
                            cur_citation = row2.Citation
                            cur_title = row2.Title
            if found_citation:
                citations.append(cur_citation)
                titles.append(cur_title)
                n_found+=1
            else:
                citations.append(None)
                titles.append(None)
        else:
            citations.append(None)
            titles.append(None)
    fp_cit_df = pd.DataFrame(df1)
    fp_cit_df['citations'] = citations
    fp_cit_df['titles'] = titles

    proportion_found = round(100*n_found/len(citations),2)
    print("Found {}% of citations in forest plots via order-agnostic, year-fuzzy and year-optional matching.".format(proportion_found))
    return fp_cit_df


def load_dfs_new(batch, file, extraction_folder='../data/extraction/'):
    dfs = {}
    try:
        dfs['citations'] = pd.read_csv(f'{extraction_folder}/parsed_citations/{batch}/{file}.csv')
    except Exception:
        dfs['citations'] = None
    try:
        dfs['fp'] = pd.read_csv(f'{extraction_folder}/FP/{batch}/FP_{file}.csv').drop('Unnamed: 0', axis=1)
    except Exception:
        dfs['fp'] = None
    try:
        dfs['rct'] = pd.read_csv(f'{extraction_folder}/RCT/{batch}/new_formatted_{file}.csv')#.drop('Unnamed: 0', axis=1)
    except Exception:
        dfs['rct'] = None
    try:
        dfs['io_tables'] = pd.read_csv(f'{extraction_folder}/io_tables/{batch}/{file}.csv')#base_name))
    except Exception:
        dfs['io_tables'] = None
    for df_type, df in dfs.items():
        if df is not None:
            df['id'] = get_ids(df, df_type)
            df['id'] = df['id'].astype(str)
            dfs[df_type] = df
            print("Found {} entries in {}".format(len(df['id']), df_type))
    return dfs


def merge_dfs(batch):
    pdf_folder = f'../data/raw_pdfs/{batch}/'
    extraction_folder = '../data/extraction/'
    out_folder = f'../data/extraction/merged_extraction/{batch}/'
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    merged_dfs = {}
    stats = {'filename':[], 'nb_rct':[], 'nb_fp':[], 'nb_citations':[], 'nb_io_tables':[], 'nb_io_text':[]}
    
    for filename in os.listdir(pdf_folder):#sub_path):
        base_name = filename.split('.')[0]
        print(base_name)
        dfs = load_dfs_new(batch, base_name, extraction_folder)
        found = True
        stats['filename'].append(base_name)
    
        found = {} #{k:v is not None for k, v in dfs.items()}
        found_rct = True
        found_fp = True
        for df_type, df in dfs.items():
            if df is None:
                found[df_type]=False
                print('Coud not find {} for SR {}'.format(df_type, base_name))
                stats['nb_{}'.format(df_type)].append(0)
            else:
                found[df_type]=True
                stats['nb_{}'.format(df_type)].append(df.shape[0])
                ## Adding id
                if df_type=='io_mentions':
                    df['study name'] = ['{} {}'.format(row.author, row.year) for i, row in df.iterrows()]
                df['id'] = get_ids(df, df_type)
                df['id'] = df['id'].astype(str)
                dfs[df_type] = df
                print("Found {} entries in {}".format(len(df['id']), df_type))
                
        if found['fp']:
            fp_cit_df = get_fp_cit_df(dfs)
            if found['rct']:
                ## Merging
                merged_df = fp_cit_df.merge(dfs['rct'], on='id', how='left')
            else:
                merged_df = fp_cit_df
            merged_df = merged_df.rename({'value':'effect size', 'year_x':'year'}, axis=1)
            if found['io_tables']:
                merged_df = merged_df.merge(dfs['io_tables'], on='id', how='left', suffixes=('', '_tables'))
            if found['rct']:
                merged_df = merged_df.drop(['type', 'year_y', 'sample size', 'group', 'programme', 'Page', 'author name'], axis=1)
                merged_df = merged_df.rename({'value':'effect size', 'year_x':'year'}, axis=1)
                reord_cols = ['id', 'titles', 'citations', 'author', 'study type', 'effect size', 'confidence interval', 'year'] + [c for c in merged_df.columns.values if c.startswith('country') or c.startswith('intervention') or c.startswith('outcome')] #, 'figure'
            else:
                reord_cols = ['id', 'titles', 'citations', 'author', 'effect size', 'confidence interval', 'year'] + [c for c in merged_df.columns.values if c.startswith('country') or c.startswith('intervention') or c.startswith('outcome')]#, 'figure'
            merged_df = merged_df[reord_cols]
            merged_df.to_csv(out_folder+'merged_extraction_{}.csv'.format(base_name))
            merged_dfs[base_name] = merged_df
    return merged_dfs, stats


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Merge extraction files for a given batch of systematic reviews."
    )

    parser.add_argument("--batch", type=str, help="Name of the batch.")

    args = parser.parse_args()

    merged_dfs, stats = merge_dfs(args.batch)

    out_path = f'data/extraction/merged_extraction/{args.batch}/'
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    for base_name, merged_df in merged_dfs.items():
        merged_df.to_csv(out_path + f'merged_extraction_{base_name}.csv'