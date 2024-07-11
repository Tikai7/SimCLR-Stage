
import torch
from model.SimCLR import SimCLR
from model.Train import Trainer
from model.Losses import NTXentLoss
from utils.DataLoaderSimCLR import DataLoaderSimCLR as DSC
from torch.utils.data import DataLoader, random_split

path_rol_comp = "../data/rol_super_compressed" 
path_sim_rol_extracted_comp = "../data/sim_rol_super_compressed" 
path_filtered = "../data/rol_super_compressed/json_filtered"
path_targets = "../data/rol_sim_rol_couples/targets.npy"

epochs = 200
image_size = 256
batch_size = 64
learning_rate = 1e-3
train_ratio = 0.8
val_ratio = 0.2
temperature = 0.5

dataset = DSC(
    path_rol_comp, path_sim_rol_extracted_comp, path_filtered, 
    shape=(image_size, image_size), target_path=path_targets, 
    sim_clr=True, use_only_rol=True
)
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

# DSC.show_data(train_loader)
# DSC.show_data(val_loader)

model = SimCLR(feature_size=128)
optimizer = torch.optim.AdamW
loss_fn = NTXentLoss(temperature=temperature)

trainer = Trainer()
trainer.set_model(model, "SimCLR-128") \
.set_optimizer(optimizer) \
.set_loss(loss_fn) 

model = trainer.fit(train_data=train_loader, validation_data=val_loader, learning_rate=learning_rate, verbose=True, epochs=epochs, sim_clr=True)

trainer.save("model_simclr.pth","history_simclr.txt")




