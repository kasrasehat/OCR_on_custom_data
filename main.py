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

dataset1 = ID_card_DataLoader(image_folder='/home/kasra/kasra_files/data-shenasname/ai_files_20230610_1',
                              label_folder='/home/kasra/kasra_files/data-shenasname/ai_metadata_20230610_1.csv',
                              transform=trans)
dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
max = 0
min = 100
k = -1
for i, data in enumerate(dataloader1):
    print(i)
    print(data[0].shape)
    for p in range(16):
        if len(list(data[1])[p]) > max:
            max = len(list(data[1])[p])

    for j in range(16):
        if len(list(data[1])[j]) < min:
            min = len(list(data[1])[j])

print(max)
print(min)

