from torch.utils.data import Dataset


class DataLoaderSimCLR(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def transform(self, X, is_target=False):
        pass

    