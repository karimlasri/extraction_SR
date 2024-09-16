from loading_utils import *
from evaluate import *
import re
import numpy as np
import ast
import pandas as pd


def get_titles(sr_annotations):
    titles = sr_annotations[sr_annotations['Field']=='Title']
    titles_dict = {}
    for i, row in titles.iterrows():
        titles_dict[row['Paper Index']] = row['Answer'].strip()
    return titles_dict


def get_citations(sr_annotations):
    citations = sr_annotations[sr_annotations['Field']=='Citation']
    citations_dict = {}
    for i, row in citations.iterrows():
        citations_dict[row['Paper Index']] = row['Answer'].strip()
    return citations_dict


def get_effect_sizes(sr_annotations):
    effect_sizes = sr_annotations[sr_annotations['Field']=='Estimate of the treatment effect']
    effect_sizes_dict = {}
    for i, row in effect_sizes.iterrows():
        paper_index = row['Paper Index']
        if paper_index in effect_sizes_dict:
            effect_sizes_dict[paper_index].append(row['Answer'].strip())
        else:
            effect_sizes_dict[paper_index] = [row['Answer'].strip()]
    return effect_sizes_dict


def get_recall_articles(fp_extraction, citations_dict):
    n_found = 0
    for paper_index, citation in citations_dict.items():
        found = False
        for i, row in fp_extraction.iterrows():
            author = row.author.split(' ')[0]
            year = row.year
            if type(author)==str and str(year) in citation:
                if citation.startswith(author):
                    found = True
                    break
        if found:
            n_found+=1
    return n_found/len(citations_dict)


def look_for_annotation(annot, citation, fp_extraction):
	found = False
	annot = annot.replace('−', '-')
	found_effect_sizes = re.findall('-?[0-9]\.[0-9]+', annot)
	if re_ann_es:
		ann_es = float(re_ann_es[0])
		re_ann_ci = re.findall('(?:\(|\[)(.*)(?:\)|\])', annot)
		if re_ann_ci:
			ann_confidence_interval = re_ann_ci[0].split(',')
			ann_ci = [float(ann_confidence_interval[0].strip()), float(ann_confidence_interval[1].strip())]
			for i, row in fp_extraction.iterrows():
				author = row.author.split(' ')[0]
				year = row.year
				if type(author)==str and str(year) in citation:
					if citation.startswith(author):
						extr_es = float(row.value)
						extr_ci = ast.literal_eval(row['confidence interval'])
						try:
							extr_ci = [float(extr_ci[0]), float(extr_ci[1])]
							if ann_es == extr_es and ann_ci == extr_ci:
								found = True
								break
						except Exception:
							pass
	return found


def get_recall_effect_sizes(fp_extraction, citations_dict, effect_sizes_dict):
    n_found = 0
    n_total = 0
    for paper_index, citation in citations_dict.items():
        if paper_index in effect_sizes_dict:
            annots = effect_sizes_dict[paper_index]
            for annot in annots:
				found = look_for_annotation(annot, citation, fp_extraction)
                if found:
                    n_found+=1
				n_total +=1
    return n_found/n_total, n_total


def evaluate_fp(annotations, fp_files):
    eval_fp = {'SR':[], 'Nb. Extractions':[], 'Recall Articles':[], 'Recall Effect Sizes':[]}
    for sr, sr_annotations in annotations.items():
        print('Evaluating forest plot extraction for SR : ', sr)
        titles_dict = get_titles(sr_annotations)
        citations_dict = get_citations(sr_annotations)
        effect_sizes_dict = get_effect_sizes(sr_annotations)
    
        if sr in fp_files:
            fp_extraction = fp_files[sr]
            n_extractions = fp_extraction.shape[0]
            es_cov, n_annotations = get_recall_effect_sizes(fp_extraction, citations_dict, effect_sizes_dict) 
            art_cov = get_recall_articles(fp_extraction, citations_dict)
        else:
            n_extractions = 0
            es_cov = 0
            art_cov = 0
            print("Extraction not found.")
        eval_fp['SR'].append(sr)
        eval_fp['Nb. Extractions'].append(n_extractions)
        eval_fp['Cov. Articles'].append(round(art_cov, 2))
        eval_fp['Cov. Effect Sizes'].append(round(es_cov, 2))
        print("Number of extractions : ", n_extractions)
        print("Number of annotations : ", sum([len(v) for k,v in effect_sizes_dict.items()]))
        print("Coverage of articles : ", art_cov)
        print("Coverage of effect sizes :", es_cov)
        print('@'*100)
    eval_fp['SR'].append('TOTAL')
    eval_fp['Nb. Extractions'].append(np.sum(eval_fp['Nb. Extractions']))
    eval_fp['Cov. Articles'].append(round(np.mean(eval_fp['Cov. Articles']), 2))
    eval_fp['Cov. Effect Sizes'].append(round(np.mean(eval_fp['Cov. Effect Sizes']), 2))
    return pd.DataFrame(eval_fp)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate effect sizes extracted from forest plots against annotations for each source pdf file."
    )

    parser.add_argument("--batch", type=str, help="Name of the batch.")

    args = parser.parse_args()

    if not os.path.exists('scores/'):
        os.mkdir('scores/')

    batch = args.batch
    
    # Load and process annotations
    sep=';'
    if batch=='A&B':
        sep=','
    annotations_folder = f'../data/annotations/{batch}/'
    annotations = load_annotations(annotations_folder, sep=sep)
    processed_annotations = {}
    for sr, sr_annot_df in annotations.items():
        processed_annotations[sr] = process_sr_annotations(sr_annot_df)

    # Load forest plots
    fp_files = {}
    fp_folder = f'../data/extraction/FP/{batch}/'
    for filename in os.listdir(fp_folder):
        if filename.endswith('.csv'):
            base_name = filename.split('.')[0].replace('FP_', '')
            fp_files[base_name] = pd.read_csv(fp_folder+filename)

    eval_fp = evaluate_fp(annotations, fp_files)
    eval_fp.to_csv(f'scores/fp_scores_{batch}.csv')