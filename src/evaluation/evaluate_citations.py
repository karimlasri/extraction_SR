from preprocess_annotations import *
from evaluate import *
import argparse
from dotenv import load_dotenv

def get_unique_citations(citations_df):
	""" Returns unique citations from a dataframe of extracted citations. """
	citations_df['id'] = get_ids(citations_df, 'citations')
	citations_unique_df = citations_df.drop_duplicates('id')
	extracted_df = citations_unique_df[~citations_unique_df['Citation'].isna()]
	unique_citations = extracted_df['Citation'].unique().tolist()
	return unique_citations


def evaluate_citations(citations_df, annotations_df):
	""" Function that evaluates citations based on the annotations for a given systematic review. """
	annotated_citations = annotations_df[annotations_df['Field']=='Citation']['Answer'].tolist()
	unique_citations = get_unique_citations(citations_df)
	intersection = 0
	extracted_authors = [name.lower().split(' ')[0].split('.')[0].split(',')[0] for name in unique_citations]
	annotated_authors = [cit.lower().split(' ')[0].split('.')[0].split(',')[0] for cit in annotated_citations]
	# Trying concurrently two matching techniques
	intersection = len(set(extracted_authors).intersection(set(annotated_authors)))
	inter = get_matches(unique_citations, annotated_citations, 20)
	intersection = max(intersection, inter)
	precision = round(intersection/len(unique_citations),2)
	recall = round(intersection/len(annotated_citations),2)
	return precision, recall
	

def evaluate_all_citations(citations_files, annotations):
	""" Function that evaluates citations based on the annotations of all systematic reviews in a batch. """
	output_path = f'scores/citations_eval_prior_{batch}.csv'
	extracted = []

	stats = {k:[] for k in ['SR', 'Precision', 'Recall']}
	for sr, citations_df in citations_files.items():
	    if sr in annotations:
	        annotations_df = annotations[sr]
	        precision, recall = evaluate_citations(citations_df, annotations_df)
	        stats['SR'].append(sr)
	        stats['Precision'].append(precision)
	        stats['Recall'].append(recall)
	        print('Precision : ', precision)
	        print('Recall : ', recall)
	stats['SR'].append('TOTAL')
	stats['Precision'].append(sum(stats['Precision'])/max(1, len(stats['Precision'])))
	stats['Recall'].append(sum(stats['Recall'])/max(1, len(stats['Recall'])))
	return stats


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate extracted citations against annotations for each source pdf file."
    )

    parser.add_argument("--batch", type=str, help="Name of the batch.")
	parser.add_argument("--outut_folder", type=str, help="The folder where the output evaluation scores should be saved.", default="scores/")

    args = parser.parse_args()

	output_path = f'{args.output_folder}/citations_eval_prior_{batch}.csv'
    
	batch = args.batch
	sep=';'
	if batch == 'A&B':
	    sep=','
	merged_folder = f'../../data/extraction/merged_extraction/{batch}/'
	merged_files = load_merged_files(merged_folder)
	
	annotations_folder = f'../../data/annotations/{batch}/'
	annotations = load_annotations(annotations_folder, sep)

	# Evaluating citations
	citations_files = {}
	citations_folder = f'../../data/extraction/parsed_citations/{batch}/'
	for filename in os.listdir(citations_folder):
	    if not filename.startswith('.'):
	        base_name = filename.split('.')[0]
	        citations_files[base_name] = pd.read_csv(citations_folder+filename)

	stats = evaluate_citations(citations_files, annotations)
	pd.DataFrame(stats).to_csv(output_path)