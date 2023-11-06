from torch.utils.data import Dataset
import cv2
import os


class UnetDataLoader(Dataset):
    def __init__(self, transformed_image_file: str, original_image_file: str, transform=None):
        super(UnetDataLoader, self).__init__()

        self.transformed_image_file = transformed_image_file
        self.original_image_file = original_image_file
        self.image_files = os.listdir(self.transformed_image_file)
        self.transform = transform


    def __getitem__(self, item):
        file_name = self.image_files[item]
        transformed_image = cv2.imread(self.transformed_image_file + f'/{file_name}')
        original_image = cv2.imread(self.original_image_file + f'/{file_name}')
        if self.transform is not None:
            transformed_image = self.transform(transformed_image)
            original_image = self.transform(original_image)

        return transformed_image, original_image


    def __len__(self):
        return len(self.image_files)
