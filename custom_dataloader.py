from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
from ast import literal_eval


class ID_card_DataLoader(Dataset):
    """
    Args:
        image_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, )
    """
    def __init__(self, image_folder: str, label_file: str, transform=None):
        super(ID_card_DataLoader, self).__init__()

        self.image_folder = image_folder
        self.label_file = label_file
        self.transform = transform
        self.labels = pd.read_csv(self.label_file, encoding='UTF-8-SIG', converters={'feature': literal_eval})
        self.image_files = os.listdir(self.image_folder)

        # Pre-filter the list of image files to only include those with valid labels
        self.valid_indices = []
        for idx, file_name in enumerate(self.image_files):
            national_id, cls = file_name.split('.')[0].split('_')
            if self.search_by_national_id(int(national_id), cls, self.labels) is not None:
                self.valid_indices.append(idx)

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

    def search_by_national_id(self, national_id, cls, dataframe):
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
        matching_row = None
        matching_row_list = dataframe[(dataframe['NATIONAL_ID'] == int(national_id)) & (dataframe['CLASS'] == int(cls))].index.tolist()
        if len(matching_row_list) == 0 and int(cls) in [2, 3]:
            if int(cls) == 2:
                cls = 3
                matching_row_list = dataframe[(dataframe['NATIONAL_ID'] == int(national_id)) & (dataframe['CLASS'] == int(cls))].index.tolist()
            elif int(cls) == 3:
                cls = 2
                matching_row_list = dataframe[(dataframe['NATIONAL_ID'] == int(national_id)) & (dataframe['CLASS'] == int(cls))].index.tolist()

        if len(matching_row_list) != 0:
            matching_row = matching_row_list[0]

        # If a match is found
        if matching_row is not None and not pd.isna(dataframe.loc[matching_row, 'PERSON_COORD']):
            # Extract the first match (assuming national IDs are unique)
            row = dataframe.iloc[matching_row]

            # Prepare the result dictionary
            result = {
                'National_ID': int(row['NATIONAL_ID']),
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
                'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
                'Class': row['CLASS'],
                'Person_coordinate': literal_eval(row['PERSON_COORD']),
                'Rotation': row['ROTATION'],
                'Scale': row['SCALE'],
                'Transport': literal_eval(row['TRANSPORT']),
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
        # Convert the birthdate to Persian numerals
        persian_birth_date = self.convert_to_persian_numbers(result['Persian Birth Date'])

        # Format the text output
        if type == 0:
            text_output = (
                f"شماره ملی: {result['National_ID']}\n"
                f"نام: {result['First Name']}\n"
                f"نام خانوادگی: {result['Last Name']}\n"
                f"تاریخ تولد: {persian_birth_date}\n"
                f"نام پدر: {result['Father Name']}"
            )
        elif type == 1:
            text_output = f"شماره سریال: {result['National ID Serial']}"
        elif type == 2:
            text_output = (
                f"نام: {result['First Name']}\n"
                f"نام خانوادگی: {result['Last Name']}\n"
                f"نام پدر: {result['Father Name']}\n"
                f"تاریخ تولد: {persian_birth_date}")

        return text_output

    def __getitem__(self, item):

        # Use the valid index
        actual_idx = self.valid_indices[item]
        file_name = self.image_files[actual_idx]
        result = self.search_by_national_id(int(file_name.split('.')[0].split('_')[0]), int(file_name.split('.')[0].split('_')[1]), self.labels)
        if result:
            try:
                image = cv2.imread(self.image_folder + '/' + file_name)
                label = self.format_result_to_persian_text(result, int(file_name.split('.')[0].split('_')[1]))
            except Exception as e:
                with open("/home/kasra/PycharmProjects/Larkimas/corrupted_files.log", "a") as log_file:
                    log_file.write(f"An error occurred while reading {file_name}: {e}\n")
                image = cv2.imread(self.image_folder + '/' + file_name)
                label = self.format_result_to_persian_text(result, int(file_name.split('.')[0].split('_')[1]))

        if result:
            if self.transform is not None:
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.valid_indices)


