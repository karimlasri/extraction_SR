from preprocess_annotations import *
import argparse
from evaluate import *
import pandas as pd

FIELDS = ['intervention', 'outcome', 'intervention_description', 'outcome_description']


def get_annot_interv(sr_annot_dict):
    annots = {field:[] for field in FIELDS}
    annot_interv = []
    annot_outcome = []
    annot_interv_desc = []
    annot_outcome_desc = []
    for paper_index, paper_annot in sr_annot_dict.items():
        for paper_field, paper_value in paper_annot.items():
            try:
                if paper_field.startswith('Interv'):
                    annots['intervention'].append(paper_value['Name of intervention'])
                    annots['intervention_description'].append(paper_value['Details of intervention'])
                    for interv_field, interv_value in paper_value.items():
                        if interv_field.startswith('Outcome'):
                            annots['outcome'].append(interv_value['Name of the outcome'])
                            if 'Outcome Details ' in interv_value.keys():
                                annots['outcome_description'].append(interv_value['Outcome Details '])
            except Exception:
                pass
    for field, values in annots.items():
        annots[field]=list(set(values))
    return annots
	

def load_io_dfs_tables(io_tables_subdir):
    io_dfs = {}
    for filename in os.listdir(io_tables_subdir):
        if filename.endswith('.csv'):
            io_df = pd.read_csv(io_tables_subdir+filename)
            io_dfs[filename[:-4]] = io_df
    return io_dfs


def evaluate_extraction_io(annot, extraction_list, eval_type='em', model_type='google'):
    sims_fields = ['annot', 'extraction', 'similarity', 'positive']
    sims_dict = {f:[] for f in sims_fields}
    if eval_type=='em':
        if annot in extraction_list:
            return 1
        else:
            return 0
    elif eval_type=='sim':
        threshold = 0.6
        emb_annot, emb_extr_list = get_embeddings(annot, extraction_list, model_type)
        for extr in extraction_list:
            sims_dict['annot'].append(annot)
            sims_dict['extraction'].append(extr)
        sims = [cosine_similarity(emb_annot, emb_extr) for emb_extr in emb_extr_list]
        sims_dict['similarity'] = [s[0][0] for s in sims]
        sims_dict['positive'] = [s[0][0]>threshold for s in sims]
        if sims:
            if max(sims) > threshold:
                return 1, sims_dict
            else:
                return 0, sims_dict
        else:
            return 0, sims_dict
            

def evaluate_io(io_dfs, processed_annotations, print_b=False, eval_type='em', model_type='google'):
    all_eval = {k:[] for k in ['sr']+['{}_{}'.format(eval_type, f) for f in FIELDS]}
    for sr, sr_io_mentions in io_dfs.items():
        sr_eval = [0, 0, 0, 0]
        total = [0, 0, 0, 0]
        annots = get_annot_interv(processed_annotations[sr])
        for i, row in sr_io_mentions.iterrows():
            for j, field in enumerate(FIELDS):
                extr = row[field]
                if str(extr)!= 'nan' and extr!='xx':
                    total[j]+=1
                    sr_eval[j]+=evaluate_extraction_io(extr, annots[field], eval_type, model_type=model_type)
        all_eval['sr'].append(sr)
        for k, field in enumerate(FIELDS):
            score = sr_eval[k]/max(1, total[k])
            all_eval['{}_{}'.format(eval_type, field)].append(score)
            if print_b:
                print('Proportion of {} for field {} : {}'.format(eval_type, field, round(score,2)))
    return all_eval


def evaluate_io2(io_dfs, processed_annotations, print_b=False, eval_type='em', model_type='google'):
    all_eval = {k:[] for k in ['sr']+['{}_{}'.format(eval_type, f) for f in FIELDS]}
    sims_fields = ['sr', 'annot', 'extraction', 'similarity', 'positive']
    all_sims_dict = {f:[] for f in sims_fields}
    for sr, sr_io_mentions in io_dfs.items():
        try:
            sr_eval = [0, 0, 0, 0]
            total = [0, 0, 0, 0]
            annots = get_annot_interv(processed_annotations[sr])
            for i, field in enumerate(FIELDS):
                for annot in annots[field]:
                    extractions = [e for e in sr_io_mentions[field].dropna().unique().tolist() if e!='xx']
                    total[i]+=1
                    evaluation, sims_dict = evaluate_extraction_io(annot, extractions, eval_type, model_type=model_type)
                    all_sims_dict['sr']+=[sr for _ in range(len(sims_dict['extraction']))]
                    for k,v in sims_dict.items():
                        all_sims_dict[k]+=v
                    sr_eval[i]+=evaluation
            all_eval['sr'].append(sr)
            for k, field in enumerate(FIELDS):
                score = sr_eval[k]/max(1, total[k])
                # print(em)
                all_eval['{}_{}'.format(eval_type, field)].append(score)
                if print_b:
                    print('Proportion of {} recall for field {} : {}'.format(eval_type, field, round(score,2)))
        except Exception:
            print('Found issue with SR {}'.format(sr))
    return all_eval, all_sims_dict


def get_scores_from_io_dfs(io_dfs, processed_annotations, print_b=False):
    all_eval = {k:[] for k in ['sr']+['em_{}'.format(f) for f in FIELDS]}
    for sr, sr_io_mentions in io_dfs.items():
        print(sr)
        sr_eval = [0, 0, 0, 0]
        total = [0, 0, 0, 0]
        try:
            annots = get_annot_interv(processed_annotations[sr])
            # print(annots)
            for i, row in sr_io_mentions.iterrows():
                for j, field in enumerate(FIELDS):
                    extr = row[field]
                    if str(extr)!= 'nan' and extr!='xx':
                        # print(field, extr)
                        total[j]+=1
                        if extr in annots[field]:
                            sr_eval[j]+=1
            # print(total, sr_eval)
            all_eval['sr'].append(sr)
            for k, field in enumerate(FIELDS):
                em = sr_eval[k]#/max(1, total[k])
                # print(em)
                all_eval['em_{}'.format(field)].append(em)
                if print_b:
                    print('Proportion EM for field {} : {}'.format(field, round(em,2)))
        except Exception:
            print('Found issue for sr {}'.format(sr))
    return all_eval


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate extracted interventions and outcomes along with their descriptions against annotations for each source pdf file."
    )

    parser.add_argument("--batch", type=str, help="Name of the batch.")
    parser.add_argument("--eval_type", type=str, help="Type of evaluation to compute recall of annotations among extraction.", default='sim')
    parser.add_argument("--save_similarities", type=str, help="A boolean indicating whether to save similarity scores.", default=True)

    args = parser.parse_args()
    
    if not os.path.exists('scores/'):
        os.mkdir('scores/')

    batch = args.batch
    eval_type = args.eval_type

    # Load and process annotations
    sep=';'
    if batch=='A&B':
        sep=','
    annotations_folder = f'../data/annotations/{batch}/'
    annotations = load_annotations(annotations_folder, sep=sep)
    processed_annotations = {}
    for sr, sr_annot_df in annotations.items():
        processed_annotations[sr] = process_sr_annotations(sr_annot_df, print_b=False)

    ## Evaluate io from mentions
    fp_folder = f'../data/extraction/FP/{batch}/'
    mentions_folder = f'../data/extraction/io_mentions/'
    mentions_subdir = mentions_folder + batch + '/'
    
    io_dfs_mentions, n_extractions = load_io_dfs_mentions(fp_folder, mentions_subdir)

	all_em, sims_dict = evaluate_io2(io_dfs_mentions, processed_annotations, print_b=False, eval_type=eval_type, model_type='GTE')
    if args.save_similarities:
        pd.DataFrame(sims_dict).to_csv(f'scores/similarities_io_mentions_{eval_type}_{batch}.csv')
    pd.DataFrame(all_em).to_csv(f'scores/recall_io_mentions_GTE_{eval_type}_{batch}.csv')


    ## Evaluate io from tables
    io_tables_folder = '../data/extraction/io_tables/' 
    io_tables_subdir = io_tables_folder + batch + '/' 
    
    io_dfs_tables = load_io_dfs_tables(io_tables_subdir)
    all_em, sims_dict = evaluate_io2(io_dfs_tables, processed_annotations, print_b=False, eval_type=eval_type, model_type='GTE')
    if args.save_similarities:
        pd.DataFrame(sims_dict).to_csv(f'scores/similarities_io_tables_{eval_type}_{batch}.csv')
    pd.DataFrame(all_em).to_csv(f'scores/recall_io_tables_GTE_{eval_type}_{batch}.csv')

    ## Evaluate io from both 
    io_dfs_concat = {}
    for sr, io_df_mentions in io_dfs_mentions.items():
        if sr in io_dfs_tables:
            io_dfs_concat[sr] = pd.concat([io_df_mentions[FIELDS], io_dfs_tables[sr][FIELDS]], axis=0)

    all_em, sims_dict = evaluate_io2(io_dfs_concat, processed_annotations, print_b=False, eval_type=eval_type, model_type='GTE')
    if args.save_similarities:
        pd.DataFrame(sims_dict).to_csv(f'scores/similarities_io_all_{eval_type}_{batch}.csv')
    pd.DataFrame(all_em).to_csv(f'scores/recall_io_all_GTE_{eval_type}_{batch}.csv')