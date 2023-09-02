from torch.utils.data import Dataset
import bdpy
import cv2
import os


class CustomDataLoader(Dataset):
    """
    Args:
        fmri_file : str :
        imagenet_folder : str
        roi_selector : str
        transform :

    Return:
        image : np.array : (N, H, W, C)
        fmri  : np.array : (N, T)
    """
    def __init__(self, fmri_file: str, imagenet_folder: str, roi_selector: str = 'ROI_VC = 1', transform=None):
        super(CustomDataLoader, self).__init__()

        self.imagenet_folder = imagenet_folder

        # Load h5 file
        fmri_data_bd = bdpy.BData(fmri_file)

        # Get ImageNet Labels
        fmri_labels = fmri_data_bd.get('Label')[:, 1].flatten()
        self.fmri_labels = ['n%08d_%d' % (int(('%f' % a).split('.')[0]),
                              int(('%f' % a).split('.')[1])) for a in fmri_labels]


        # Get fMRI data in the ROI
        self.fmri_data = fmri_data_bd.select(roi_selector)

        self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(self.imagenet_folder + f'/{self.fmri_labels[item]}.JPEG')
        fmri = self.fmri_data[item]

        if self.transform is not None:
            image = self.transform(image)

        return image, fmri

    def __len__(self):
        return len(self.fmri_labels)


class ID_card_DataLoader(Dataset):
    """
    Args:
        imagenet_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, 1)
    """
    def __init__(self, image_folder: str, transform=None):
        super(ID_card_DataLoader, self).__init__()

        self.image_folder = image_folder
        self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(self.image_folder + f'/{os.listdir(self.image_folder)[item]}')
        label = 1

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(os.listdir(self.coco_folder))


class Custom_real_DataLoader(Dataset):
    """
    Args:
        imagenet_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, 1)
    """
    def __init__(self, coco_folder: str, transform=None):
        super(Custom_real_DataLoader, self).__init__()

        self.coco_folder = coco_folder
        self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(self.coco_folder + f'/{os.listdir(self.coco_folder)[item]}')
        label = 1

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(os.listdir(self.coco_folder))