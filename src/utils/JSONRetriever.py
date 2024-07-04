
import os
import requests
from tqdm import tqdm
import json
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

    @staticmethod
    def filter_json(path, path_link):  
        """
        Filter JSON files.
        @param path: The path to the json files.
        @param path_link: The path to the images files that we want to link with the json files.
        """
        images_sim = set(map(lambda x: x.split('_')[0], os.listdir(path_link)))
        for file in os.listdir(path+'/json'):
            try:
                data = json.load(open(f"{path}/json/{file}", encoding='utf-8'))
                for obj in data['metadata']:
                    if obj['label'] == 'Relation' and obj['value'].split('/')[-1] in images_sim:
                        with open(f"{path}/json_filtered/{file}", "w", encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        break
            except:
                continue    
                
                
