import google.generativeai as genai
import pandas as pd
import json
import os
import re
import argparse


def load_citations(pdf_name, citations_folder):
    path = "{}{}.csv".format(citations_folder, pdf_name.split(".")[0])
    if os.path.exists(path):
        citations = pd.read_csv(path)
        return citations["Citation"].tolist()
    else:
        return []


def get_citation_information(citation, model):
    messages = [
        f"""You are an assistant that extracts paper titles, first author names
            and year in citations for economics RCT/impact evaluation studies.
            Extract a JSON containing the title, author names and year
            from the following citation.
            If there are no author names, output 'xx'.
            Only output author names and year found in the text.
            If there are no author names or year present, output 'xx'.

            JSON Format:
            {{"Title":"xx", "Authors":"xx", "Year":"xx"}}

            Citation: {citation}

            Please provide the results in JSON format.

            Json:"""
    ]

    response = model.generate_content(messages)
    response.resolve()
    return response.text


def process_citations(citations):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    data = []
    for citation in citations:
        try:
            citation_info = get_citation_information(citation, model)
            info_as_dict = json.loads(re.findall("(\{.*\})", citation_info)[0])
            info_as_dict["Citation"] = citation
            data.append(info_as_dict)
        except Exception:
            pass
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse extracted citations using google's Gemini model."
    )

    parser.add_argument("--pdf_folder", type=str, help="Path to the input pdfs.")
    parser.add_argument(
        "--citations_folder", type=str, help="Path to the extracted citations."
    )
    parser.add_argument("--out_folder", type=str, help="Path to output.")

    args = parser.parse_args()

    pdf_folder = "data/Batch_Aug_6"
    citations_folder = args.citations_folder
    out_folder = args.out_folder

    if not os.path.exists(args.out_folder):
        os.mkdir(args.out_folder)

    for pdf_name in os.listdir(args.pdf_folder):
        if pdf_name.endswith(".pdf"):
            citations = load_citations(pdf_name, args.citations_folder)
            # Process captions
            data = process_citations(citations)
            # Convert to DataFrame
            df = pd.DataFrame(data)
            # Save to CSV
            df.to_csv(args.out_folder + pdf_name.split(".")[0] + ".csv", index=False)
