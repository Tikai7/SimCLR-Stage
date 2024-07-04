
import os
import requests
from tqdm import tqdm
class JSONRetriever:
    """
    Class to retrieve images from JSON files.
    """
    @staticmethod
    def get_json_from_images(path, dest_path):
        """
        Get JSON files from images.
        @param path: The path to the images files.
        @param dest_path: The path to save the json.
        """
        all_computed_json = set(os.listdir(dest_path))
        print(f"Already computed {len(all_computed_json)} json files.")
        for img in tqdm(os.listdir(path)):
            try:
                img_id = img.split('.')[0]
                if img_id+".json" in all_computed_json:
                    continue 
                json_file = requests.get(f"https://gallica.bnf.fr/iiif/ark:/12148/{img_id}/manifest.json")
                with open(f"{dest_path}/{img_id}.json", "w", encoding='utf-8') as f:
                    f.write(json_file.text) 
            except:
                continue