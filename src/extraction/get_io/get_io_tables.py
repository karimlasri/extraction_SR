import pickle as pkl
import os
import re
import openai
import json
import google.generativeai as genai
import argparse


patterns = ['intervention', 'outcome', '(description|details)', '(author|name)']

def load_tables(tables_folder):
    all_tables = {}
    for subdir_name in os.listdir(tables_folder):
        base_name = subdir_name.split('.')[0]
        subdir_path = tables_folder + subdir_name + '/'
        sr_tables = []
        for filename in os.listdir(subdir_path):
            file_path = subdir_path + filename
            if filename.endswith('.pkl'):
                sr_tables.append(pkl.load(open(file_path, 'rb')))
                
        all_tables[base_name] = sr_tables
    return all_tables


def is_io_table(table):
    is_io = False
    for c in table.columns.values:
        for p in patterns[:3]:
            if p in str(c).lower():
                is_io = True
    return is_io


def get_candidate_col_names_re(all_tables):
    col_names = {p:[] for p in patterns}
    n_ios = {}
    n_found = [0 for _ in patterns]
    for sr, sr_tables in all_tables.items():
        n_io = 0
        for t in sr_tables:
            latex_str = t.reset_index().to_latex()
            if is_io_table(t):
                n_io +=1
                for i, concept in enumerate(patterns):
                    found = False
                    for c in t.columns.values:
                        if len(re.findall(concept, str(c).lower()))>0:
                            if print_log:
                            col_names[concept].append(c)
                            found = True
                            n_found[i]+=1
        n_ios[sr] = n_io
    return col_names, n_ios, n_found


def transform_study_name(study_name):
    year_candidates = re.findall('[0-9]{4}', study_name)
    if year_candidates:
        year_suffix = '_'+year_candidates[0]
    else:
        year_suffix = ''
    return study_name.split(' ')[0].lower()+year_suffix


def create_index(df):
    study_names = df['study name']
    return [transform_study_name(study_name) for study_name in study_names]



def get_io_from_latex_with_gpt(latex_content):
	""" Extract information about interventions and outcomes from a LaTeX table using GPT. """
    # Preprocess the LaTeX content to get only the table part
    table_pattern = re.compile(r'\\begin{tabular}.*?\\end{tabular}', re.DOTALL)
    table_match = table_pattern.search(latex_content)

    # If no table found, raise an error
    if not table_match:
        raise ValueError("No table found in the LaTeX content")

    table_content = latex_content
    # Escape curly braces in the LaTeX content
    escaped_table_content = table_content.replace('{', '{{').replace('}', '}}')

    # Prepare the prompt for GPT-4
    prompt = f"""
    Given a LaTeX table entry giving information for impact evaluation studies.
    Look for specific keywords and phrases that indicate the table is about interventions and/or outcomes. 
    Extract the intervention, outcome and their description from the
    following latex table along with the study name.
    The output should be a dictionary with five keys : 'study name', 'intervention', 'outcome' \
    'intervention_description' and 'outcome_description', respectively for the intervention, outcome, the \
    description of the intervention, and the description of the outcome. If any of these four fields is not \
    found, the associated value should be xx. Ensure the well-formedness of the output JSON. \
    
    JSON Format:
    {{"study name": "xx", "intervention": "xx", "outcome": "xx", "intervention_description": "xx", "outcome_description": "xx"}}    
    
    Latex: \n {escaped_table_content}"

    

    Please provide the results in JSON format.

    Json:
    """

    # Call the OpenAI API
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts interventions and outcomes \
        along with their description in latex tables from economics RCT/impact evaluation studies.."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0
    )

    # Extract the classifications from the response
    classifications = response.choices[0].message.content.strip()
    return classifications


def get_io_from_latex_with_google(latex_content, model):
	""" Extract information about interventions and outcomes from a LaTeX table using Gemini. """
    # Preprocess the LaTeX content to get only the table part
    table_pattern = re.compile(r'\\begin{tabular}.*?\\end{tabular}', re.DOTALL)
    table_match = table_pattern.search(latex_content)

    # If no table found, raise an error
    if not table_match:
        raise ValueError("No table found in the LaTeX content")

    table_content = latex_content
    # Escape curly braces in the LaTeX content
    escaped_table_content = table_content.replace('{', '{{').replace('}', '}}')

    # Prepare the prompt for Google Gemini
    prompt = f"""
    You are an assistant that extracts information about interventions, outcomes along with their description, and countries in latex tables from
    economics RCT/impact evaluation studies. You are given a LaTeX table as an input, which contains information for impact evaluation studies. 
    Look for specific keywords and phrases that indicate the table is about interventions and/or outcomes. Extract the intervention, outcome and 
    their description from the following latex table along with the study name. The output should be a dictionary with six keys : 'study name', 
    'intervention', 'outcome', 'intervention_description', 'outcome_description', and 'country' respectively for the name of the study (author 
    and eventually year), the intervention, the name of the outcome, the description of the intervention, the description of the outcome, and the 
    country where the study was implemented. If any of these four fields is not found, the associated value should be xx. Assert the well-formedness 
    of the output JSON. 
    
    JSON Format:
    {{"study name": "xx", "intervention": "xx", "outcome": "xx", "intervention_description": "xx", "outcome_description": "xx", "country": "xx"}}    
    
    Latex: \n {escaped_table_content}"

    

    Please provide the results in JSON format.

    Json:
    """
    
    response = model.generate_content(prompt)
    response.resolve()
    return response.text


if __name__=='__main__':
	parser = argparse.ArgumentParser(
        description="Extract information about interventions and outcomes from tables."
    )

    parser.add_argument("--tables_folder", type=str, help="Path to the input tables.")
    parser.add_argument("--out_folder", type=str, help="Path to output.")

    args = parser.parse_args()
	
	tables_folder = args.table_folder 
	all_tables = load_tables(tables_folder)
	
	col_names, n_ios, n_found = get_candidate_col_names_re(all_tables)
	
	genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
	model = genai.GenerativeModel('gemini-1.5-flash')
	
	
	n_io = 0
	n_found = [0 for _ in range(5)]
	
	out_folder = args.out_folder
	
	if not os.path.exists(out_folder):
	    os.mkdir(out_folder)


	for sr, sr_tables in all_tables.items():
	    extracted_jsons = []
	    for t in sr_tables:
	        try:
	            latex_str = t.reset_index().to_latex()
	            if is_io_table(t):
	                n_io +=1
	                io_response = get_io_from_latex_with_google(latex_str, model)
	                io_response = io_response.replace('\n', ' ')
	                json_as_list_candidates = re.findall('\[.*\]', io_response)
	                json_as_dict_candidates = re.findall('\{.*\}', io_response)
	                
	                if json_as_dict_candidates:
	                    as_list=False
	                    if json_as_list_candidates:
	                        if len(json_as_list_candidates[0])>len(json_as_dict_candidates[0]):
	                            as_list=True
	                            io_as_json = json.loads(json_as_list_candidates[0])
	                            for io_dict in io_as_json:
	                                extracted_jsons.append(io_dict)
	                                for i, k in enumerate(io_dict.keys()):
	                                    if io_dict[k] != 'xx':
	                                        n_found[i]+=1
	                    if as_list==False:
	                        io_as_dict = json.loads(json_as_dict_candidates[0])
	                        extracted_jsons.append(io_as_dict)
	                        for i, k in enumerate(io_as_dict.keys()):
	                                if io_as_dict[k] != 'xx':
	                                    n_found[i]+=1
	        except Exception:
	            pass
	    try:
	        df = pd.DataFrame(extracted_jsons)
	        df['id'] = create_index(df)
	        df.to_csv(out_folder+sr+'.csv', index=False)
	    except Exception:
	        pass