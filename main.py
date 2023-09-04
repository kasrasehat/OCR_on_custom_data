from custom_dataloader import ID_card_DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import pickle
import tqdm

height, width = 224, 224
batch_size = 16
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

files = glob.glob('/home/kasra/kasra_files/data-shenasname/*.CSV') + glob.glob('/home/kasra/kasra_files/data-shenasname/*.csv')
file_list = []
for file in files:
    file_list.append([file, file.split('.')[0].replace('metadata', 'files')])
tokens = []
max = 0
min = 100

for index, file in tqdm.tqdm(enumerate(file_list)):

    dataset1 = ID_card_DataLoader(image_folder=file[1], label_folder=file[0], transform=trans)
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    for i, data in tqdm.tqdm(enumerate(dataloader1)):
        print(i)
        print(data[0].shape)
        for p in range(16):
            if len(list(data[1])[p]) > max:
                max = len(list(data[1])[p])

        for j in range(16):
            if len(list(data[1])[j]) < min:
                min = len(list(data[1])[j])

        for k in range(len(list(data[1]))):
            for token in list(list(data[1])[k]):
                if token not in tokens:
                    tokens.append(token)



print(max)
print(min)
with open("tokens_list.pkl", "wb") as f:
    pickle.dump(tokens, f)
