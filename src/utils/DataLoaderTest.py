import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DataLoaderTest(Dataset):

    def __init__(self, path_to_sim_test="",shape=(256,256)) -> None:
        super().__init__()
        self.all_files = os.listdir(path_to_sim_test)
        self.all_files = sorted(self.all_files, key=lambda x:int(x.split("ID_")[1].split('.')[0]))
        self.shape = shape 
        self.path_sim_test = path_to_sim_test
        self.transform = transforms.Compose([
                transforms.Resize(self.shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.target_names = []
        self.image_names = []
        for i in range(0,len(self.all_files)-1,2):
            target = self.all_files[i]
            img = self.all_files[i+1]

            if not img.startswith("btv"):
                img, target = target, img

            self.target_names.append(target)
            self.image_names.append(img)

        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_file = self.image_names[index]
        target_file = self.target_names[index]

        img = Image.open(f"{self.path_sim_test}/{img_file}").convert("L")
        target = Image.open(f"{self.path_sim_test}/{target_file}").convert("L")

        img = self.transform(img)
        target = self.transform(target)

        return img, target
