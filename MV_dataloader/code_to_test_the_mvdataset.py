from torchvision import transforms
from torch.utils.data import DataLoader
from MVImage_Net_dataloader import MVImage_Net_Dataset
from MVImage_net_Sampling import SamplingDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = SamplingDataset(root_path='/data/rxm210041/mvi_00', transform=transform, mode='labeled', num_classes=40)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
batch = next(iter(data_loader))