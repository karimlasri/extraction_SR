# Requirements

- Python >= 3.9
- Git

# Configuration
1. **Clone the repository**

    Use the following command to clone the repository:
    ```bash
    git clone <repository-url>
    ```

2. **Install poetry**

    Poetry is a tool for dependency management in Python. Install it with the following command:

    ```bash
    curl -sSL https://install.python-poetry.org | python -
    ```

3. **Install dependencies**

    Use Poetry to install the project dependencies:

    ```bash
    poetry install
    ```

4. **Configure pre-commit**

    Pre-commit is a tool that runs checks on your code before you commit it. It is configured in the `.pre-commit-config.yaml` file. To install it, run the following command:

    ```bash
    poetry run pre-commit install
    ```

5. **Set-up environment variables**

    Create a `.env` file at the root of the project and add the following environment variables
        - OPENAI_API_KEY
        - HUGGINGFACE_API_KEY

    And activate the variables with:

    ```bash
    source .env
    ```

# Data structure
Most scripts in this repository are written to perform extraction subsequent transforms on a batch of pdfs. 
As a suggestion, prior to running the extraction pipeline, systematic review files could be put as a batch in a new subfolder of the ```data/raw_pdfs``` folder, e.g. ```data/raw_pdfs/Aug_6/```. The ```data``` folder should also contain an extraction subfolder, ```data/extraction``` where the output of extraction scripts will be stored as a batch subfolder, e.g. ```data/extraction/RCT```. 
The last suggestion is to also include annotations in a subfolder such as ```data/annotations/Aug_6```. 


# Extraction pipeline

## Extract and parse bibliography

### Bibliography extraction

This scripts aims to extract all relevant bibliography entries from source pdfs.

To execute this script, run:

```bash
python src/extraction/get_bibliography/get_citations.py --pdf_folder <path_to_pdfs> --out_folder <path_to_extractions> 
```

With:
- `pdf_folder`: the path to the input pdfs for systematic reviews considered.
- `out_folder`: the path to output folder where extractions will be saved as a csv file.

For example:

```bash
python src/extraction/get_bibliography/get_citations.py --pdf_folder data/raw_pdfs/Batch_Aug_6/ --out_folder get_citations/extracted_citations/citations_Aug_6/
```

The output folder will contain csv files corresponding to input pdf files, each containing the extracted citations for a given systematic review.

### Bibliography parsing

This scripts aims to parse the bibliography strings previously extracted from source pdfs.

To run the parsing with Google Gemini, run:

```bash
python src/extraction/get_bibliography/parse_citations_google.py --pdf_folder <path_to_pdfs> --citations_folder <path_to_extractions> --out_folder <path_to_extractions> 
```

With:
- `pdf_folder`: the path to the input pdfs for systematic reviews considered.
- `citations_folder`: the path to the extracted citations.
- `out_folder`: the path to output folder where extractions will be saved as a csv file.

For example:

```bash
python src/extraction/get_bibliography/parse_citations_google.py --pdf_folder data/Batch_Aug_6/ --citations_folder get_citations/extracted_citations/citations_Aug_6/ --out_folder get_citations/parsed_citations/parsed_citations_Aug_6/
```

The output folder will contain csv files corresponding to input pdf files, each containing the parsed citations for a given systematic review.


## RCT Extraction		

   Use the following command to run the `main_rct.py` script:

   ```bash
   python src/extraction/get_rct/main_rct.py --pdf_path <path_to_your_pdf> --output_folder <output_folder_path>
   ```

   Replace <path_to_your_pdf> with the path to your input PDF file.
   Replace <output_folder_path> with the desired output folder where images and CSV files will be saved.

    Example Command:

    ```bash
    python src/extraction/get_rct/main_rct.py --pdf_path data/pdf/A2.pdf --output_folder data/RCT
    ```

    Script Workflow
    1. The script will process the specified PDF file and extract RCTness.
    2. Extracted data will be saved as images (metadata) and in a CSV file in the specified output folder.
    3. Ensure the output folder exists or will be created automatically in the script.


## Forest Plot Extraction

   Use the following command to run the `main_FP.py` script:

   ```bash
   python src/extraction/get_forest_plot/main_FP.py --pdf_path <path_to_your_pdf> --output_folder <output_folder_path>
   ```

   Replace <path_to_your_pdf> with the path to your input PDF file.
   Replace <output_folder_path> with the desired output folder where images and CSV files will be saved.

    Example Command:

    ```bash
    python src/extraction/get_forest_plot/main_FP.py --pdf_path data/pdf/A1.pdf --output_folder data/ForestPlots
    ```

    Script Workflow
    1. The script will process the specified PDF file and extract Forest Plot data.
    2. Extracted data will be saved as images (metadata) and in a CSV file in the specified output folder.
    3. Ensure the output folder exists or will be created automatically in the script.


## Table Extraction

   Use the following command to run the `main_table.py` script:

   ```bash
   python src/extraction/get_tables/main_table.py --pdf_path <path_to_your_pdf> --output_folder <output_folder_path>
   ```

   Replace <path_to_your_pdf> with the path to your input PDF file.
   Replace <output_folder_path> with the desired output folder where images and CSV files will be saved.

    Example Command:

    ```bash
    python src/extraction/get_tables/main_table.py --pdf_path data/pdf/A1.pdf --output_folder data/TableExtraction
    ```

    Script Workflow
    1. The script will process the specified PDF file and extract Tables.
    2. Extracted data will be saved as images (metadata) and in a .pkl files in the specified output folder.
    3. Ensure the output folder exists or will be created automatically in the script.


## Extraction of interventions and outcomes from tables

This script aims to get information about interventions and outcomes (their names and descriptions) from extracted tables.

To run the parsing with Google Gemini, run:

```bash
python src/extraction/get_io/get_io_tables.py --tables_folder <path_to_tables> --out_folder <path_to_extraction> 
```

With:
- `tables_folder`: the path to the input tables for the systematic reviews considered.
- `out_folder`: the path to output folder where extractions will be saved as a csv file.

For example:

```bash
python src/extraction/get_io/get_io_tables.py --tables_folder data/extraction/tables/ --out_folder data/extraction/io_tables/ 
```

The output folder will contain csv files corresponding to input pdf files, each containing the interventions, outcomes and descriptions for a given systematic review.


# Files merging

This script is aimed at merging files previously extracted from source pdfs.

To run the merging, you should first place extraction files in the following data subfolders:
- `data/extraction/FP` for forest plots,
- `data/extraction/RCT` for RCT-ness,
- `data/extraction/io_tables` for extractions from tables,
- `data/extraction/parsed_citations` for the parsed bibliography.

```bash
python src/merge_files/merge_files.py --batch <name_of_batch> 
```

Where `batch` is the name of the batch from which the information is extracted

For example:

```bash
python src/merge_files/merge_files.py --batch Aug_6
```

The output folder will by default be `data/extraction/merged_extraction/{batch}` and will contain a csv file merging all the information extracted for each raw pdf.


# Evaluation

These scripts are aimed at evaluating files previously extracted from source pdfs against annotation files.

To run the evaluation, you should first run the merging described in the section above.

## RCTness evaluation

This script evaluates the extracted RCTness in both merged files and original files prior to merging. It should be ran as follows :

```bash
python src/evaluation/evaluate_rct.py --batch Aug_6
```

The output folder will by default be `data/evaluation/scores/` and will contain csv files containing the evaluation scores.

## Intervention-outcome evaluation

This script evaluates the extracted interventions and outcomes in both mentions from the text and tables prior to merging. It should be ran as follows :

```bash
python src/evaluation/evaluate_io.py --batch Aug_6 --eval_type sim --save_similarities True
```

The output folder will by default be `data/evaluation/scores/` and will contain csv files containing the evaluation scores, as well as pairwise similarities if the `save_similarities` argument was set to `True`.

## Effect size evaluation

This script evaluates the effect sizes extracted from forest plots prior to merging. It should be ran as follows :

```bash
python src/evaluation/evaluate_fp.py --batch Aug_6
```

The output folder will by default be `data/evaluation/scores/` and will contain csv files containing the evaluation scores.


## Evaluate citations

This script evaluates the extracted citations in original files prior to merging. It should be ran as follows :

```bash
python src/evaluation/evaluate_citations.py --batch Aug_6 --output_folder scores/
```

The output folder will by default be `data/evaluation/scores/` and will contain a csv files containing the evaluation scores.
