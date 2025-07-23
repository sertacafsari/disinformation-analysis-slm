import json
import csv
import os
import re
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from huggingface_hub import login
from dotenv import load_dotenv
import requests
from io import BytesIO
from PIL import Image as PILImage
from tqdm import tqdm


def convertCSVToJSON(csv_path:str, json_path:str):
    """ Converts the CSV file to JSON file and saves it """
    with open(csv_path, newline='', encoding='utf-8') as csv_f:
        reader = csv.DictReader(csv_f)
        rows = list(reader)
    
    with open(json_path, 'w', encoding='utf-8') as json_f:
        json.dump(rows, json_f, indent=2, ensure_ascii=False)

def cleanFaux():
    """ A function to clean the Fauxtography data for multimodal experiments"""
    
    convertCSVToJSON("./data/datasets/faux/snopes.csv", "./data/datasets/faux/snopes.json")
    with open("./data/datasets/faux/snopes.json", 'r', encoding='utf-8') as json_f:
        snopes_json = json.load(json_f)

    # Ensure there is not any missing value in claim, img_main, and label columns
    snopes_json = [item for item in snopes_json if item.get('claim') and item.get('img_main') and item.get('label')]

    # Remove unnecessary keys and values from the dataset
    snopes_json = [{key: value for key, value in item.items() if key in ['claim', 'img_main', 'label']} for item in snopes_json]

    with open("./data/datasets/faux/snopes_processed.json", 'w', encoding='utf-8') as json_f:
        json.dump(snopes_json, json_f, indent=2, ensure_ascii=False)
    
    # Image downloading
    updated_data = []

    for idx, item in enumerate(tqdm(snopes_json, desc="Downloading Images")):

        image_url = item["img_main"]
        image_path = f"./data/datasets/faux/images/image_{idx}.jpg"

        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            res = requests.get(image_url,headers=headers)
            res.raise_for_status()

            with PILImage.open(BytesIO(res.content)) as img:
                img.convert("RGB").save(image_path)
            
            item["img_main"] = image_path
            updated_data.append(item)
        except requests.exceptions.RequestException as e:
            print("Image link is broken")
            continue

    # Generate a Huggingface data
    hf_data = Dataset.from_list(updated_data)

    # Get class labels
    unique_labels = sorted(hf_data.unique("label"))
    class_labels = ClassLabel(names=unique_labels)

    # Split the dataset into training, test
    train_test_set = hf_data.train_test_split(test_size=0.4, seed=42)
    train_data = train_test_set["train"]

    # Split test to validation and test splits
    test_validation_set = train_test_set["test"]

    val_test = test_validation_set.train_test_split(test_size=0.5,seed=42)
    validation_data = val_test["train"]
    test_data = val_test["test"]

    # Generate a final dataset
    final_data = DatasetDict(
        {
            "train": train_data,
            "validation": validation_data,
            "test": test_data
        }
    ).cast_column("label", class_labels)

    final_data.save_to_disk("./data/datasets/faux/")
    final_data["train"].to_json("./data/datasets/faux/train_processed.json")
    final_data["validation"].to_json("./data/datasets/faux/validation_processed.json")
    final_data["test"].to_json("./data/datasets/faux/test_processed.json")

    return final_data


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

    hf_data = Dataset.from_list(cleaned_data)

    unique_labels = sorted(hf_data.unique("label"))

    class_labels = ClassLabel(names=unique_labels)

    # Split the dataset into training, test
    train_test_set = hf_data.train_test_split(test_size=0.4, seed=42)
    train_data = train_test_set["train"]

    # Split test to validation and test splits
    test_validation_set = train_test_set["test"]

    val_test = test_validation_set.train_test_split(test_size=0.5,seed=42)
    validation_data = val_test["train"]
    test_data = val_test["test"]

    # Combine the data
    final_data = DatasetDict(
        {
            "train": train_data,
            "validation": validation_data,
            "test": test_data
        }
    ).cast_column("label", class_labels)

    final_data.save_to_disk("./data/datasets/argilla/")
    final_data["train"].to_json("./data/datasets/argilla/train_processed.json")
    final_data["validation"].to_json("./data/datasets/argilla/validation_processed.json")
    final_data["test"].to_json("./data/datasets/argilla/test_processed.json")

    return final_data

if __name__ == "__main__":
    faux = cleanFaux()
    argilla = cleanArgilla()