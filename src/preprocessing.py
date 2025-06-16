import json
import csv
import os
import re
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv


def convertCSVToJSON(csv_path:str, json_path:str):
    """ Converts the CSV file to JSON file and saves it """
    with open(csv_path, newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        rows = list(reader)
    
    with open(json_path, 'w', encoding='utf-8') as json_f:
        json.dump(rows, json_f, indent=2, ensure_ascii=False)

def cleanFaux():
    """ A function to clean the Fauxtography data for multimodal experiments"""
    convertCSVToJSON("./data/datasets/faux/reuters.csv", "./data/datasets/faux/reuters.json")
    with open("./data/datasets/faux/reuters.json", 'r', encoding='utf-8') as json_f:
        reuters_json = json.load(json_f)
    
    convertCSVToJSON("./data/datasets/faux/snopes.csv", "./data/datasets/faux/snopes.json")
    with open("./data/datasets/faux/snopes.json", 'r', encoding='utf-8') as json_f:
        snopes_json = json.load(json_f)

    # Ensure there is not any missing value in claim, img_main, and label columns
    reuters_json = [item for item in reuters_json if item.get('claim') and item.get('img_main') and item.get('label')]
    snopes_json = [item for item in snopes_json if item.get('claim') and item.get('img_main') and item.get('label')]

    # Remove unnecessary keys and values from the dataset
    reuters_json = [{key: value for key, value in item.items() if key in ['claim', 'img_main', 'label']} for item in reuters_json]
    snopes_json = [{key: value for key, value in item.items() if key in ['claim', 'img_main', 'label']} for item in snopes_json]

    with open("./data/datasets/faux/reuters_processed.json", 'w', encoding='utf-8') as json_f:
        json.dump(reuters_json, json_f, indent=2, ensure_ascii=False)

    with open("./data/datasets/faux/snopes_processed.json", 'w', encoding='utf-8') as json_f:
        json.dump(snopes_json, json_f, indent=2, ensure_ascii=False)
     
    # Return 395 Reuters data, and 838 Snopes data
    return reuters_json, snopes_json



def cleanArgilla():
    """ Clean the Argilla dataset for textual experiments """

    # Login to Huggingface
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)

    argilla = load_dataset("argilla/news-fakenews")

    # Remove unnecessary columns from the dataset
    removed_data = argilla.remove_columns(["inputs", "prediction_agent", "annotation", "annotation_agent", "multi_label", "explanation", "id", "metadata", "status", "event_timestamp", "metrics"])["train"]

    # PLACE (SOURCE) pattern
    data_source_pattern = re.compile(r'^.*?\s*\([^)]+\)\s*-\s*')

    cleaned_data = []
    for item in removed_data.to_list():
        # Remove PLACE (SOURCE) pattern as it provides the insight whether the news is real or not
        if 'text' in item and item['text']:
            original = item['text']
            cleaned_text = data_source_pattern.sub('', original).strip()
            item['text'] = cleaned_text
        
        # Remove the "prediction" and just have label key and its corresponding value
        if 'prediction' in item and item['prediction']:
            info = item['prediction'][0]
            if 'label' in info:
                item['label'] = info['label']
        
        # Ensure prediction is removed
        if 'prediction' in item:
            del item['prediction']

        cleaned_data.append(item)
    
    # Delete the first data entry as it is null
    del cleaned_data[0]

    # Save the processed Argilla data
    with open("./data/datasets/argilla/argilla_processed.json", "w", encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)


    return cleaned_data

if __name__ == "__main__":
    r,s = cleanFaux()
    argilla = cleanArgilla()