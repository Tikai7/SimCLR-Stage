import os 
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

class CaptionRetriever():
    def __init__(self) -> None:
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def retrieve_caption_for(self,path):
        computed_images = set(os.listdir(f"{path}/captions"))
        print(f"[INFO] Already computed {len(computed_images)} files")
        all_images = os.listdir(path)
        
        for image_file in tqdm(all_images):
            try: 
                if image_file.replace(".jpg",".txt") in computed_images:
                    continue
                image = Image.open(f"{path}/{image_file}")
                inputs = self.processor(images=image, return_tensors="pt")
                inputs.to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**inputs)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                with open(f"{path}/captions/{image_file.replace('.jpg','.txt')}", "w") as f :
                    f.write(caption)
            except Exception as e:
                print(e)
                continue
