import google.generativeai as genai
import pandas as pd
import json
import os
import re
import argparse
import openai

from dotenv import load_dotenv  # Import dotenv to load .env file

load_dotenv()

# Few-shot examples for guiding the model with a variety of author formats - parse citations
few_shot_examples = """
Example 1:
Citation: "Smith, J. (2020). The impact of education on economic growth. Journal of Economics."
Json: {"Title": "The impact of education on economic growth", "Authors": "Smith", "Year": "2020"}

Example 2:
Citation: "Doe, A. (2019). Microfinance and poverty alleviation. Economic Review."
Json: {"Title": "Microfinance and poverty alleviation", "Authors": "Doe", "Year": "2019"}

Example 3:
Citation: "World Bank. (2021). Global economic prospects. Washington, DC."
Json: {"Title": "Global economic prospects", "Authors": "World Bank", "Year": "2021"}

Example 4:
Citation: "UNICEF. (2018). Children and the digital divide. UNICEF Publications."
Json: {"Title": "Children and the digital divide", "Authors": "UNICEF", "Year": "2018"}

Example 5:
Citation: "Johnson, B., Lee, C., & Miller, D. (2022). Healthcare access in rural areas. Health Economics Journal."
Json: {"Title": "Healthcare access in rural areas", "Authors": "Johnson", "Year": "2022"}

Example 6:
Citation: "OECD. (2017). Education at a Glance. OECD Publishing."
Json: {"Title": "Education at a Glance", "Authors": "OECD", "Year": "2017"}

Example 7:
Citation: "Greenpeace. (2020). The state of the climate crisis. Greenpeace Reports."
Json: {"Title": "The state of the climate crisis", "Authors": "Greenpeace", "Year": "2020"}

Example 8:
Citation: "Adams, P., & Kim, J. (2016). Economic reforms in emerging markets. Journal of Development Studies."
Json: {"Title": "Economic reforms in emerging markets", "Authors": "Adams", "Year": "2016"}

Example 9:
Citation: "Gates Foundation. (2015). Improving health outcomes in Africa. Gates Foundation Publications."
Json: {"Title": "Improving health outcomes in Africa", "Authors": "Gates Foundation", "Year": "2015"}

Example 10:
Citation: "Nguyen, T., Patel, S., & Brown, K. (2020). Small businesses and economic resilience. Economic Policy."
Json: {"Title": "Small businesses and economic resilience", "Authors": "Nguyen", "Year": "2020"}
"""


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


# Function to extract citation information using OpenAI GPT-4 API with few-shot examples
def get_citation_information_openai(citation):
    """Function to get parsed citations for economics RCT studies from their 'raw' string."""
    # Define messages for the chat model, including few-shot examples
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that extracts the paper title, first author's last name, or organization name if no individual author is listed, "
                "and the year from citations for economics RCT/impact evaluation studies. Extract a JSON containing the title, author names, "
                "and year from the following citation. If there are no author names, output the organization name. "
                "Only output the last name of the first author found in the text. \n\n"
                'JSON Format:\n{"Title":"xx", "Authors":"xx", "Year":"xx"}\n\n'
                f"Few-shot examples:\n{few_shot_examples}\n\n"
                f"Citation: {citation}\n\n"
                "Please provide the results in JSON format.\n\nJson:"
            ),
        }
    ]

    # Using OpenAI's GPT-4 chat model to generate the response
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use GPT-4 model identifier
        messages=messages,
        max_tokens=500,
        temperature=0,
    )

    # Extract the response text
    raw_response = response.choices[0].message["content"].strip()

    # Clean up the response to extract the JSON object
    json_matches = re.findall(r"\{.*?\}", raw_response, re.DOTALL)
    if json_matches:
        # Return the first valid JSON object found
        return json_matches[0]
    else:
        raise ValueError(f"Failed to extract JSON from response: {raw_response}")


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


# Function to process multiple citations using GPT-4 API
def process_citations_openai(citations):
    # Ensure OpenAI API key is set in your environment
    openai.api_key = os.getenv(
        "OPENAI_API_KEY"
    )  # Set your OpenAI API key in the environment

    data = []
    for citation in citations:
        try:
            # Get citation information from GPT-4o
            citation_info = get_citation_information_openai(citation)

            # Parse the cleaned JSON response
            info_as_dict = json.loads(citation_info)

            # Add the original citation text to the dictionary
            info_as_dict["Citation"] = citation

            # Append parsed information to the data list
            data.append(info_as_dict)
        except json.JSONDecodeError:
            print(
                f"Failed to parse JSON for citation: '{citation}'. Response was: {citation_info}"
            )
        except Exception as e:
            # Print or log the error if needed
            print(f"Error processing citation '{citation}': {e}")
            continue

    return data


# Main function to process all PDFs in the folder and output results to CSV
def process_pdfs(pdf_folder, citations_folder, out_folder):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for pdf_name in os.listdir(pdf_folder):
        if pdf_name.endswith(".pdf"):
            citations = load_citations(pdf_name, citations_folder)
            if citations:
                # Process the loaded citations
                data = process_citations_openai(citations)

                # Convert the processed data into a DataFrame
                df = pd.DataFrame(data)

                # Save the DataFrame to CSV in the output folder
                output_path = os.path.join(out_folder, pdf_name.split(".")[0] + ".csv")
                df.to_csv(output_path, index=False)


# Example usage
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

    # Set the paths
    pdf_folder = args.pdf_folder
    citations_folder = args.citations_folder
    out_folder = args.out_folder

    process_pdfs(pdf_folder, citations_folder, out_folder)

"""
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
"""
