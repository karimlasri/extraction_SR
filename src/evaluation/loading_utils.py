import os
import pandas as pd

LEVELS_DICT = {
    'paper' : ['1. Title and publication details​', '4. Topics and Objectives', '5. Sampling'],
    'intervention' : ['6.Intervention'],
    'outcome' : ['7.Outcome', '8.Estimates']
}


def load_annotations(annotations_folder, sep=','):
    annotations = {}
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.csv'):
            base_name = filename.split('.')[0]
            annotations[base_name] = pd.read_csv(annotations_folder+filename, sep=sep)
    return annotations


def load_merged_files(merged_folder):
    merged_files = {}
    for filename in os.listdir(merged_folder):
        if filename.endswith('.csv'):
            base_name = filename.split('.')[0].replace('merged_extraction_', '')
            merged_files[base_name] = pd.read_csv(merged_folder+filename)
    return merged_files


def process_paper_annot(sr_annot_df, paper_index, print_b=False):
    if print_b:
        print('Processing paper index : {}'.format(int(paper_index)))
    paper_annot_dict = {}
    paper_annot_df = sr_annot_df[sr_annot_df['Paper Index']==paper_index]
    paper_level_annot_df = paper_annot_df[paper_annot_df['Topic'].isin(LEVELS_DICT['paper'])]
    # Get paper level fields
    for field in paper_level_annot_df['Field']:
        paper_annot_dict[field] = paper_level_annot_df[paper_level_annot_df['Field']==field]['Answer'].tolist()[0]
    # Get lower levels
    intervention_col_name = [c for c in paper_annot_df.columns.values if c.startswith('Intervention')][0]
    for interv_index in paper_annot_df[intervention_col_name].dropna().unique():
        paper_annot_dict['Interv_{}'.format(interv_index)] = process_interv_annot(paper_annot_df, interv_index)
    return paper_annot_dict
    

def process_interv_annot(paper_annot_df, interv_index, print_b=False):
    if print_b:
        print('Processing intervention index : {}'.format(int(interv_index)))
    interv_annot_dict = {}
    intervention_col_name = [c for c in paper_annot_df.columns.values if c.startswith('Intervention')][0]
    interv_annot_df = paper_annot_df[paper_annot_df[intervention_col_name]==interv_index]
    interv_level_annot_df = interv_annot_df[interv_annot_df['Topic'].isin(LEVELS_DICT['intervention'])]
    # Get intervention level fields
    for field in interv_level_annot_df['Field']:
        interv_annot_dict[field] = interv_level_annot_df[interv_level_annot_df['Field']==field]['Answer'].tolist()[0]
    # Get lower levels
    for outcome_index in interv_annot_df['Outcome Index'].dropna().unique():
        interv_annot_dict['Outcome_{}'.format(outcome_index)] = process_outcome_annot(interv_annot_df, outcome_index)
    return interv_annot_dict


def process_outcome_annot(interv_annot_df, outcome_index, print_b=False):
    if print_b:
        print('Processing outcome index : {}'.format(outcome_index))
    outcome_annot_dict = {}
    outcome_annot_df = interv_annot_df[interv_annot_df['Outcome Index']==outcome_index]
    outcome_level_annot_df = outcome_annot_df[outcome_annot_df['Topic'].isin(LEVELS_DICT['outcome'])]
    # Get intervention level fields
    for field in outcome_level_annot_df['Field']:
        outcome_annot_dict[field] = outcome_level_annot_df[outcome_level_annot_df['Field']==field]['Answer'].tolist()[0]
    return outcome_annot_dict


def process_sr_annotations(sr_annot_df, print_b=False):
    sr_annot_dict = {}
    for paper_index in sr_annot_df['Paper Index'].dropna().unique():
        sr_annot_dict['Paper_{}'.format(paper_index)] = process_paper_annot(sr_annot_df, paper_index, print_b)
    return sr_annot_dict