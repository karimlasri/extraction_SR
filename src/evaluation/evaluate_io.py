from loading_utils import *
import argparse
from evaluate import *
import pandas as pd

FIELDS = ['intervention', 'outcome', 'intervention_description', 'outcome_description']


def get_annot_interv(annotations_df):
    annots = {field:annotations_df[annotations_df['Field']==field]['Answer'].tolist() for field in FIELDS}
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


def look_for_io_annotation(annot, extraction_list, eval_type='em', model_type='google'):
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


def evaluate_io_for_sr(sr_annotations, sr_io_mentions, eval_type='em', model_type='google'):
	""" Evaluates extraction for one systematic review """
	sr_eval = [0] * len(FIELDS)
	total = [0] * len(FIELDS)
	annots = get_annot_interv(sr_annotations)
	for i, field in enumerate(FIELDS):
		for annot in annots[field]:
			extractions = [e for e in sr_io_mentions[field].dropna().unique().tolist() if e!='xx']
			total[i]+=1
			evaluation, sims_dict = look_for_io_annotation(annot, extractions, eval_type, model_type=model_type)
			all_sims_dict['sr']+=[sr for _ in range(len(sims_dict['extraction']))]
			for k,v in sims_dict.items():
				all_sims_dict[k]+=v
			sr_eval[i]+=evaluation
	return sr_eval, total


def evaluate_io(io_dfs, annotations, print_b=False, eval_type='em', model_type='google'):
	""" Evaluates extraction for all systematic reviews """
    all_eval = {k:[] for k in ['sr']+['{}_{}'.format(eval_type, f) for f in FIELDS]}
    sims_fields = ['sr', 'annot', 'extraction', 'similarity', 'positive']
    all_sims_dict = {f:[] for f in sims_fields}
    for sr, sr_io_mentions in io_dfs.items():
        try:
            sr_eval, total = evaluate_io_for_sr(annotations[sr], sr_io_mentions, eval_type, model_type)
            all_eval['sr'].append(sr)
            for k, field in enumerate(FIELDS):
                score = sr_eval[k]/max(1, total[k])
                all_eval['{}_{}'.format(eval_type, field)].append(score)
                if print_b:
                    print('Proportion of {} recall for field {} : {}'.format(eval_type, field, round(score,2)))
        except Exception:
            print('Found issue with SR {}'.format(sr))
    return all_eval, all_sims_dict


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

    ## Evaluate io from tables
    io_tables_folder = '../data/extraction/io_tables/' 
    io_tables_subdir = io_tables_folder + batch + '/' 
    
    io_dfs_tables = load_io_dfs_tables(io_tables_subdir)
    all_em, sims_dict = evaluate_io(io_dfs_tables, annotations, print_b=False, eval_type=eval_type, model_type='google')
    if args.save_similarities:
        pd.DataFrame(sims_dict).to_csv(f'scores/similarities_io_tables_{eval_type}_{batch}.csv')
    pd.DataFrame(all_em).to_csv(f'scores/recall_io_tables_GTE_{eval_type}_{batch}.csv')
