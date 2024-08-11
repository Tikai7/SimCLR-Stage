
import torch
from model.SimCLR import SimCLR
from model.Train import Trainer
from model.Losses import NTXentLoss, ContrastiveLoss
from utils.DataLoaderSimCLR import DataLoaderSimCLR as DSC
from torch.utils.data import DataLoader, random_split



path_rol_ht_super_comp = "../data/rol_ht_super_compressed" 
path_rol_comp = "../data/rol_super_compressed" 
path_sim_rol_extracted_comp = "../data/sim_rol_super_compressed" 
path_filtered = "../data/rol_super_compressed/json_filtered"
path_sim_rol_test = "../data/sim_rol_test"
path_targets = "../data/rol_sim_rol_couples/targets.npy"
bad_pairs_path = "./files/bad_pairs.txt"
to_enhance_path = "./files/to_enhance_pairs.txt"

epochs = 30
image_size = 256
batch_size = 64
learning_rate = 1e-4
train_ratio = 0.8
val_ratio = 0.2
temperature = 0.5

dataset = DSC(
    path_rol_comp, path_sim_rol_extracted_comp, path_filtered, 
    shape=(image_size, image_size), target_path=path_targets,
    to_enhance_path=to_enhance_path, bad_pairs_path=bad_pairs_path,
    path_sim_rol_test=path_sim_rol_test, max_images=40000,
    augment_test=False, use_only_rol=True, use_context=True, remove_to_enhance_files=True, remove_bad_pairs=True
)

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size 

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


model = SimCLR(feature_size=128, use_context=True, context_weights=1.0)
optimizer = torch.optim.AdamW
loss_fn = ContrastiveLoss(temperature=temperature)

trainer = Trainer()
trainer.set_model(model, "SimCLR-RGC") \
.set_optimizer(optimizer) \
.set_loss(loss_fn) 

model = trainer.fit(train_data=train_loader, validation_data=val_loader,
                     learning_rate=learning_rate, verbose=True, epochs=epochs, sim_clr=True, use_context=True)

trainer.save("model_simclr_RGC.pth","history_simclr_RGC.txt")
