from loading_utils import *
from evaluate import *
import os


def evaluate_rctness_merged(merged_files, annotations):
    rct_stats = {k:[] for k in ['SR', 'Precision', 'Recall']}
    for sr, merged_df in merged_files.items():
        print("Computing RCT metrics for : ", sr)
        if sr, sr_annotations in annotations.items():
            annotated_titles = sr_annotations[sr_annotations['Field']=='Title']['Answer'].tolist()
            
            merged_unique_df = merged_df.drop_duplicates('id')
            if 'study type' in merged_unique_df.columns.values:
                na_ratio = sum(merged_unique_df['study type'].isna())/merged_unique_df['study type'].shape[0]
                print('Proportion not extracted : ', na_ratio)
                merged_unique_df = merged_unique_df[~merged_unique_df['study type'].isna()]
            extracted_titles = merged_unique_df['titles'].astype(str).tolist()
        #     dists = get_lev(annotated_titles, extracted_titles)
            inter = get_matches(annotated_titles, extracted_titles)
            precision = inter/max(1, len(extracted_titles))
            recall = inter/max(1, len(annotated_titles))
            print('Precision : ', precision)
            print('Recall : ', recall)
            rct_stats['SR'].append(sr)
            rct_stats['Precision'].append(precision)
            rct_stats['Recall'].append(recall)
        else:
            print(f'WARNING : {sr} not found in annotations.')
        print('-'*50)
    rct_stats['SR'].append('TOTAL')
    rct_stats['Precision'].append(sum(rct_stats['Precision'])/len(rct_stats['Precision']))
    rct_stats['Recall'].append(sum(rct_stats['Recall'])/len(rct_stats['Recall']))
    return pd.DataFrame(rct_stats)


def evaluate_rctness_prior(rct_files, annotations):
    rct_stats = {k:[] for k in ['SR', 'Precision', 'Recall']}
    for sr, rct_df in rct_files.items():
        print("Computing RCT metrics for : ", sr)
        if sr, sr_annotations in annotations.items():
            annotated_citations = sr_annotations[sr_annotations['Field']=='Citation']['Answer'].tolist()
            rct_df['id'] = get_ids(rct_df, 'rct')
            rct_unique_df = rct_df.drop_duplicates('id')
            extracted_df = rct_unique_df[~rct_unique_df['study type'].isna()]
            extracted_authors = extracted_df[extracted_df['study type']=='RCT']['author name'].unique().tolist()
            
            intersection = 0
            for name in extracted_authors:
                found = False
                for cit in annotated_citations:
                    if name.lower().split(' ')[0] == cit.lower().split(' ')[0].split('.')[0].split(',')[0]:
                        found = True
                if found:
                    intersection+=1
        
            precision = round(intersection/max(1, len(extracted_authors)),2)
            recall = round(intersection/max(1, len(annotated_citations)),2)
            rct_stats['SR'].append(sr)
            rct_stats['Precision'].append(precision)
            rct_stats['Recall'].append(recall)
            print('Precision : ', precision)
            print('Recall : ', recall)
        else:
            print(f'WARNING : {sr} not found in annotations.')
        print('-'*50)
    rct_stats['SR'].append('TOTAL')
    rct_stats['Precision'].append(sum(rct_stats['Precision'])/len(rct_stats['Precision']))
    rct_stats['Recall'].append(sum(rct_stats['Recall'])/len(rct_stats['Recall']))
    return pd.DataFrame(rct_stats)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate extracted RCT-ness against annotations for each source pdf file."
    )

    parser.add_argument("--batch", type=str, help="Name of the batch.")

    args = parser.parse_args()
    
    if not os.path.exists('scores/'):
        os.mkdir('scores/')

    batch = args.batch
    sep=';'
    if batch=='A&B':
        sep=','

    # Load merged files and annotations
    merged_folder = f'../data/extraction/merged_extraction/{batch}/'
    merged_files = load_merged_files(merged_folder)
    
    annotations_folder = f'../data/annotations/{batch}/'
    annotations = load_annotations(annotations_folder, sep)

    df_stats_merged = evaluate_rctness_merged(merged_files, annotations)
    df_stats_merged.to_csv(f'scores/RCT_stats_merged_{batch}.csv')

    # Load RCT files prior to merging
    rct_files = {}
    rct_folder = f'../data/extraction/RCT/{batch}/'
    for filename in os.listdir(rct_folder):
        if filename.endswith('.csv'):
            sr = filename.split('.')[0].replace('new_formatted_', '')
            rct_files[sr] = pd.read_csv(rct_folder+filename)

    df_stats_prior = evaluate_rctness_prior(rct_files, annotations)
    df_stats_prior.to_csv(f'scores/RCT_stats_prior_{batch}.csv')
    