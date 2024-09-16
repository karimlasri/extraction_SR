import os
import pandas as pd


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

