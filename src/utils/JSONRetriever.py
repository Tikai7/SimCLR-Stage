import re
import os
import requests
from tqdm import tqdm
import json
import random
from textattack.augmentation import WordNetAugmenter, EasyDataAugmenter
 
augmenter = WordNetAugmenter()
eda_augmenter = EasyDataAugmenter()

class JSONRetriever:
    """
    Class to retrieve images from JSON files.
    """
    @staticmethod
    def get_captions(image_file, path, target=False, augment=False, caption_folder="detailed-captions"):          
        """
        Get captions from the images
        @param image_file : name of the image
        @param path : path of the image
        @param target : if the image is going to be the target (in this case the image is in the sim_rol folder)
        @param augment : if we want to augment the text
        """

        try:
            image_file = image_file.split('\\')[1].replace('.jpg','.txt') if target else image_file+'.txt'
            with open(f"{path}/{caption_folder}/{image_file}", "r") as f:
                text = f.read()
            
            text = text.split('. ')
            text = '. '.join(text[:1])

            if augment:
                # print(text)
                text = augmenter.augment(text)[0]
                text = eda_augmenter.augment(text)
                text = text[random.randint(0,len(text)-1)]
                # print(text)
                
            return text
        except Exception as e:
            print(f"[ERROR] {e}")
            return "<UNK>"

    @staticmethod
    def get_encoded_context(model, path, path_json, target=False, folder_root="json_filtered"):
        """
        Get context from the images within the JSON files
        @param path : path of the image
        @param target : if the image is going to be the target, in this case we'll check the sim_rol folder
        Context is built from : 
            - title
            - date
        """
        try:
            json_path = None
            if target:
                ark = re.findall('bpt.*', path)[0].split('_')[0]
                json_path = path_json+f'/json/{ark}.json'
            else:
                json_path = path_json+f'/{folder_root}/{path}.json'
            data = json.load(open(json_path,"r",encoding="utf-8"))
            text = ""
            for obj in data['metadata']:
                if obj['label'] == "Title" or obj['label'] == "Date":
                    text += f"{obj['value']} "
                    
            encoded_text = model(text).squeeze(0)
            return encoded_text
        except Exception as e:
            return None

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
        for file in tqdm(os.listdir(path+'/json')):
            try:
                data = json.load(open(f"{path}/json/{file}", encoding='utf-8'))
                for obj in data['metadata']:
                    found = False
                    if obj['label'] == 'Relation':
                        arks = re.findall("bpt.*", str(obj))
                        for ark in arks:
                            ark_id = re.sub("/.*", ".json", ark).split('.')[0]
                            if ark_id in images_sim:        
                                found = True                        
                                with open(f"{path}/json_filtered/{file}", "w", encoding='utf-8') as f:
                                    json.dump(data, f, ensure_ascii=False, indent=4)
                                break
                        if found:
                            break
            except:
                continue    
    
    @staticmethod
    def get_all_relations(path_filered_json):
        """
        Get all the relations from the filtered json files.
        @param path_filered_json: The path to the filtered json files.
        """
        liste_ark = []
        liste_triplet = {}

        for file in os.listdir(path_filered_json):
            file_id = file.split(".")[0]
            data = json.load(open(path_filered_json + "/" + file, "r", encoding="utf-8"))
            for obj in data['metadata']:
                if obj['label'] == "Relation":
                    arks = re.findall("12148\/b.*", str(obj)) # bpt
                    for ark in arks :
                        liste_ark.append(ark.replace("'}", "").split('/')[1])
                        if file_id not in liste_triplet:
                            liste_triplet[file_id] = [ark.replace("'}", "")]
                        else:
                            liste_triplet[file_id].append(ark.replace("'}", ""))

        return liste_ark, liste_triplet
            
    @staticmethod
    def assert_filtered_json(path_json_sim, path_json_filtered):
        """
        Assert the filtered json files.
        @param path_json_sim: The path to the json sim files.
        @param path_json_filtered: The path to the json filtered files.
        """

        liste_ark, _ = JSONRetriever.get_all_relations(f'{path_json_filtered}')
        json_sim = os.listdir(f'{path_json_sim}')
        cpt = 0
        for file in json_sim:
            if file.split(".")[0] in set(liste_ark):
                cpt += 1
        print(f"Number of json files in json_sim: {len(json_sim)}, number of json files in json_filtered: {cpt}")
        # assert cpt == len(json_sim), f"Error: {cpt} != {len(json_sim)}"
                        
                
