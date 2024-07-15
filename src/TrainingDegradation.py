
import torch
import segmentation_models_pytorch as smp
from model.Train import Trainer
from torch.utils.data import DataLoader, random_split
from utils.DataLoaderSimCLR import DataLoaderSimCLR as DSC

epochs = 50
image_size = 256
num_classes = 1
batch_size = 32
learning_rate = 1e-4
train_ratio = 0.8
val_ratio = 0.2


path_rol_comp = "../data/rol_super_compressed" 
path_sim_rol_extracted_comp = "../data/sim_rol_super_compressed" 
path_filtered = "../data/rol_super_compressed/json_filtered"
path_targets = "../data/rol_sim_rol_couples/targets.npy"


dataset = DSC(
    path_rol_comp, path_sim_rol_extracted_comp, path_filtered, 
    shape=(image_size, image_size), target_path=path_targets, 
    augment_test=False, use_only_rol=False, use_context=False
)

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


# DSC.show_data(train_loader)
# DSC.show_data(val_loader)


encoder_name = 'efficientnet-b1'
model = smp.Unet(encoder_name=encoder_name, encoder_weights='imagenet', classes=num_classes, in_channels=3)
optimizer = torch.optim.AdamW
loss_fn = torch.nn.MSELoss()

trainer = Trainer()
trainer.set_model(model, encoder_name) \
.set_optimizer(optimizer) \
.set_loss(loss_fn) 

model = trainer.fit(train_data=train_loader, validation_data=val_loader, learning_rate=learning_rate, verbose=True, epochs=epochs)

trainer.save("model_degradation.pth","history_degradation.txt")
