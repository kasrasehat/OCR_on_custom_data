from custom_dataloader import ID_card_DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader

height, width = 224, 224
batch_size = 16
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

dataset1 = ID_card_DataLoader(image_folder='/home/kasra/kasra_files/data-shenasname/ai_files_20230528',
                              label_folder='/home/kasra/kasra_files/data-shenasname/ai_metadata_20230528.csv',
                              transform=trans)
dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
for i, data in enumerate(zip(dataloader1)):
    print(i)

