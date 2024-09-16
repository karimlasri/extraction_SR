import os
import fitz  # PyMuPDF
import torch
import argparse
from paddleocr import PaddleOCR
import openai

from utils import (
    initialize_table_models,
    find_pages_with_tables,
    find_pages_with_keywords,
    create_output_folder,
    save_pdf_pages_as_images,
)
from config.keywords_config import KEYWORDS
from generate_utils import main, process_dictionary_to_csv
from dotenv import load_dotenv  # Import dotenv to load .env file

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


print(api_key)


def process_pdf(
    pdf_path, output_folder, reference_range, detection_thresholds, crop_padding, device
):
    """Main function to process the PDF and extract table data."""
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    # Open the PDF document
    doc = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)

    base_name = os.path.splitext(pdf_name)[0]
    temp_output_folder = os.path.join(output_folder, base_name)
    os.makedirs(temp_output_folder, exist_ok=True)

    # Initialize table models
    model, structure_model = initialize_table_models()

    # Find pages with tables and keywords
    pages_with_tables = find_pages_with_tables(doc)
    pages_with_rct = find_pages_with_keywords(doc, KEYWORDS)

    # Find intersection of pages containing tables and keywords
    intersection = list(set(pages_with_tables) & set(pages_with_rct))
    intersection.sort()
    # Save relevant pages as images
    output_folder_image_path = create_output_folder(
        os.path.join(temp_output_folder, "Images")
    )
    save_pdf_pages_as_images(pdf_path, intersection, output_folder_image_path)

    # Filter out reference pages if reference range is provided
    if reference_range:
        references = list(range(min(reference_range), max(reference_range) + 1))
        intersection = list(set(intersection) - set(references))
        intersection.sort()

    # Extract data from the processed pages
    Dictionary = main(
        pdf_path,
        intersection,
        detection_thresholds,
        output_folder_image_path,
        model,
        structure_model,
        device,
        crop_padding,
        ocr,
        temp_output_folder,
    )

    # Process the dictionary into a DataFrame and save as CSV
    final_df = process_dictionary_to_csv(Dictionary)

    final_df.to_csv(f"{temp_output_folder}/new_formatted_{base_name}.csv", index=False)
    print("Processing complete. Output saved to:", f"new_formatted_{base_name}.csv")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a PDF to extract tables and relevant data."
    )
    parser.add_argument(
        "--pdf_path", type=str, required=True, help="Path to the PDF file."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/RCT",
        help="Output folder for images and results.",
    )
    parser.add_argument(
        "--reference_range",
        type=int,
        nargs=2,
        help="Range of reference pages to exclude (start, end). Optional.",
    )
    parser.add_argument(
        "--table_threshold",
        type=float,
        default=0.85,
        help="Threshold for detecting tables.",
    )
    parser.add_argument(
        "--rotated_table_threshold",
        type=float,
        default=0.5,
        help="Threshold for detecting rotated tables.",
    )
    parser.add_argument(
        "--no_object_threshold",
        type=int,
        default=10,
        help="Threshold for detecting no object.",
    )
    parser.add_argument(
        "--crop_padding", type=int, default=10, help="Padding for table crops."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the models on (cuda or cpu).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Define detection thresholds
    detection_thresholds = {
        "table": args.table_threshold,
        "table rotated": args.rotated_table_threshold,
        "no object": args.no_object_threshold,
    }

    # Call the main processing function
    process_pdf(
        pdf_path=args.pdf_path,
        output_folder=args.output_folder,
        reference_range=args.reference_range,  # Pass None if not provided
        detection_thresholds=detection_thresholds,
        crop_padding=args.crop_padding,
        device=args.device,
    )
