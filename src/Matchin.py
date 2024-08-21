
from torch.utils.data import DataLoader
from model.SimCLR import SimCLR
from model.Train import Trainer
from utils.DataLoaderTest import DataLoaderTest as DLT
from utils.Similarity import Similarity as SMY
from utils.Plotter import Plotter as PL


path_sim_rol_test = "src/data/data_PPTI/sim_rol_test"
batch_size = 32

model = SimCLR(feature_size=128)
model_state = Trainer().get_model("C:/Cours-Sorbonne/M1/Stage/src/params/model/model_simclr_RGC.pth")
model.load_state_dict(model_state)
history = Trainer().get_history("C:/Cours-Sorbonne/M1/Stage/src/params/model/history_simclr_RGC.txt")

testset = DLT(path_to_sim_test=path_sim_rol_test, augment=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

top_k_pairs, original_images, augmented_images, precisions = SMY.match_images_with_simCLR(
    model, test_loader=test_loader, k=10, use_sift=False, is_test=True, alpha=0.6
)

best_pairs = top_k_pairs[:,0]
PL.plot_best_pairs(best_pairs, original_images, augmented_images, max_images=5)