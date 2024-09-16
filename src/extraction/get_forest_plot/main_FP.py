from paddleocr import PaddleOCR
import openai
import argparse
import fitz
import os
from generate_utils import (
    main_Forest,
    process_extracted_captions,
    process_and_save_images,
    preprocess_document_and_features,
    extract_and_process_captions,
    process_data,
)

from caption import process_document_for_captions
from dotenv import load_dotenv  # Import dotenv to load .env file


# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


# Ensure the API key is available
if not api_key:
    raise ValueError("OpenAI API key is missing. Please set it in the .env file.")


# Define your main processing function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process PDF and extract captions.")
    parser.add_argument(
        "--pdf_path", type=str, required=True, help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/ForestPlots",
        help="Output folder for images and CSV.",
    )
    return parser.parse_args()


def main(args):
    pdf_path = args.pdf_path
    output_folder_path = args.output_folder

    doc = fitz.open(pdf_path)

    df_new = extract_and_process_captions(pdf_path)

    df_extracted_rule_features = process_document_for_captions(
        doc, "config/clf_parsing_logs.pkl"
    )

    new_df_extracted_rule_features, df_new = preprocess_document_and_features(
        df_extracted_rule_features, df_new
    )

    df_new["caption_lowercase"].unique()

    df_extracted_captions = process_extracted_captions(
        new_df_extracted_rule_features, df_new
    )

    df_new = df_extracted_captions.copy()
    page_numbers = list(df_new["Page"].unique())
    page_numbers.sort()

    process_and_save_images(
        page_numbers, pdf_path, output_folder_path, output_folder_name="Images"
    )

    Dictionary = {}

    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    main_dict = main_Forest(
        page_numbers, pdf_path, df_new, api_key, Dictionary, ocr, args.output_folder
    )
    df_sorted, extracted_string = process_data(main_dict, pdf_path)

    df_sorted.to_csv(f"{output_folder_path}/{extracted_string}.csv")


"""
def main():
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    main_dict = main_Forest(page_numbers, pdf_path, df_new, api_key, Dictionary, ocr)
    df_sorted, extracted_string = process_data(main_dict, pdf_path)

    df_sorted.to_csv(f"data/ForestPlots/{extracted_string}.csv")
"""

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
