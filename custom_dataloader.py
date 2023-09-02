from torch.utils.data import Dataset
import cv2
import os
import pandas as pd


class ID_card_DataLoader(Dataset):
    """
    Args:
        imagenet_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, 1)
    """
    def __init__(self, image_folder: str, label_folder: str, transform=None):
        super(ID_card_DataLoader, self).__init__()

        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.labels = pd.read_csv(self.label_folder)
        self.image_files = os.listdir(self.image_folder)

    @staticmethod
    def convert_to_persian_numbers(arabic_str):

        """
        Convert a string containing Arabic numerals to Persian numerals.

        Parameters:
            arabic_str (str): The string containing Arabic numerals.

        Returns:
            str: The string with Arabic numerals replaced by Persian numerals.
        """
        arabic_to_persian = {
            '0': '۰',
            '1': '۱',
            '2': '۲',
            '3': '۳',
            '4': '۴',
            '5': '۵',
            '6': '۶',
            '7': '۷',
            '8': '۸',
            '9': '۹'
        }

        return ''.join(arabic_to_persian.get(char, char) for char in arabic_str)

    # Modify the search function to convert Persian Birth Date to Persian numbers

    def search_by_national_id(self, national_id, dataframe):
        """
        Search for a given national ID in the dataframe and return other columns' values and index if found.

        Parameters:
            national_id (int or str): The national ID to search for.
            dataframe (pd.DataFrame): The DataFrame to search in.

        Returns:
            dict: A dictionary containing the other columns' values and index if the national ID is found.
            None: If the national ID is not found.
        """
        # Search for the national ID in the 'NATIONAL_ID' column
        match = dataframe[dataframe['NATIONAL_ID'] == int(national_id)]

        # If a match is found
        if not match.empty:
            # Extract the first match (assuming national IDs are unique)
            row = match.iloc[0]

            # Prepare the result dictionary
            result = {
                'National_ID': row['NATIONAL_ID'],
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
                'National ID Serial': row['NATIONAL_ID_SERIAL'],
                'Index': row.name  # row.name contains the index
            }

            return result
        else:
            return None

    # Define a function to convert the result dictionary to a formatted Persian text output
    def format_result_to_persian_text(self, result, type):
        """
        Format the result dictionary to a Persian text output.

        Parameters:
        - result: A dictionary containing the data, index, and total_count (as returned by find_info_by_national_id function).

        Returns:
        - A string containing the formatted text output in Persian.
        """
        if result['data'] is None:
            return None

        # Convert the birthdate to Persian numerals
        persian_birth_date = self.convert_to_persian_numbers(result['data']['PERSIAN_BIRTH_DATE'])

        # Format the text output
        if type == 0:
            text_output = (
                f"شماره ملی: {result['data']['National_ID']}\n"
                f"نام: {result['data']['First Name']}\n"
                f"نام خانوادگی: {result['data']['Last Name']}\n"
                f"تاریخ تولد: {persian_birth_date}\n"
                f"نام پدر: {result['data']['Father Name']}"
            )
        elif type == 1:
            text_output = f"شماره سریال: {result['data']['NATIONAL_ID_SERIAL']}"
        elif type == 2:
            text_output = (
                f"نام: {result['data']['FIRST_NAME']}\n"
                f"نام خانوادگی: {result['data']['LAST_NAME']}\n"
                f"نام پدر: {result['data']['FATHER_NAME']}\n"
                f"تاریخ تولد: {persian_birth_date}")

        return text_output

    def __getitem__(self, item):

        result = self.search_by_national_id(self.image_files[item].split('.')[0].split('_')[0], self.labels)
        if result:
            image = cv2.imread(self.image_folder + f'/{os.listdir(self.image_folder)[item]}')
            label = self.format_result_to_persian_text(result, self.image_files[item].split('.')[0].split('_')[1])

        if result:
            if self.transform is not None:
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(os.listdir(self.image_folder))


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

