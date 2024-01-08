from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
from ast import literal_eval
from tokenizer import char2ind
import numpy as np
import torch


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
        self.invalid_indices = []
        for idx, file_name in enumerate(self.image_files):
            national_id, cls = file_name.split('.')[0].split('_')
            if self.search_by_national_id(int(national_id), cls, self.labels) is not None:
                self.valid_indices.append(idx)
            else:
                self.invalid_indices.append(idx)

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
            a = str(int(row['NATIONAL_ID']))
            while len(list(a)) != 10:
                a = '0' + a

            if row['SCALE'] == 0:
                scale = 1
            else:
                scale = row['SCALE']

            result = {
                'National_ID': a,
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
                'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
                'Class': int(row['CLASS']),
                'Person_coordinate': np.round(np.array(literal_eval(row['PERSON_COORD'])).reshape(8)/720, 2),
                'Rotation': (row['ROTATION'] + 1)/2,
                'Scale': scale,
                'Transport': np.round(((np.array(literal_eval(row['TRANSPORT'])).reshape(2)/360) +
                             np.ones_like(np.array(literal_eval(row['TRANSPORT'])).reshape(2)))/2, 2),
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
            text_output = {'passage': (
                                f"a\n"
                                f"شماره ملی: {result['National_ID']}\n"
                                f"نام: {result['First Name']}\n"
                                f"نام خانوادگی: {result['Last Name']}\n"
                                f"تاریخ تولد: {persian_birth_date}\n"
                                f"نام پدر: {result['Father Name']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}

        elif type == 1:
            text_output = {'passage': (f"b\n"
                                       f"{result['National ID Serial']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 2:
            text_output = {'passage': (
                                f"c"),
                                # f"نام: {result['First Name']}\n"
                                # f"نام خانوادگی: {result['Last Name']}\n"
                                # f"نام پدر: {result['Father Name']}\n"
                                # f"تاریخ تولد: {persian_birth_date}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 3:
            text_output = {'passage': (
                f"d"),
                # f"نام: {result['First Name']}\n"
                # f"نام خانوادگی: {result['Last Name']}\n"
                # f"نام پدر: {result['Father Name']}\n"
                # f"تاریخ تولد: {persian_birth_date}"),
                'class': result['Class'],
                'person_coordinate': result['Person_coordinate'],
                'rotation': result['Rotation'],
                'scale': result['Scale'],
                'transport': result['Transport']}

        return text_output

    def __getitem__(self, item):

        # Use the valid index
        MAX_LENGTH = 160
        EOS_token = 1
        actual_idx = self.valid_indices[item]
        file_name = self.image_files[actual_idx]
        result = self.search_by_national_id(int(file_name.split('.')[0].split('_')[0]), int(file_name.split('.')[0].split('_')[1]), self.labels)
        target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127
        if result:
            try:
                image = cv2.imread(self.image_folder + '/' + file_name)
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids
            except Exception as e:
                with open("E:/codes_py/Larkimas/hs_err_pid11016.log", "a") as log_file:
                    log_file.write(f"An error occurred while reading {file_name}: {e}\n")
                image = cv2.imread(self.image_folder + '/' + file_name)
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids

        if result:
            if self.transform is not None:
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.valid_indices)




class All_ID_card_DataLoader(Dataset):
    """
    Args:
        image_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, )
    """
    def __init__(self, data_file: str, transform=None):
        super(All_ID_card_DataLoader, self).__init__()

        self.data_file = data_file
        self.all_data = pd.read_csv(self.data_file)
        self.transform = transform

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
            a = str(int(row['NATIONAL_ID']))
            while len(list(a)) != 10:
                a = '0' + a

            if row['SCALE'] == 0:
                scale = 1
            else:
                scale = row['SCALE']

            result = {
                'National_ID': a,
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
                'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
                'Class': int(row['CLASS']),
                'Person_coordinate': np.round(np.array(literal_eval(row['PERSON_COORD'])).reshape(8)/720, 2),
                'Rotation': (row['ROTATION'] + 1)/2,
                'Scale': scale,
                'Transport': np.round(((np.array(literal_eval(row['TRANSPORT'])).reshape(2)/360) +
                             np.ones_like(np.array(literal_eval(row['TRANSPORT'])).reshape(2)))/2, 2),
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
            text_output = {'passage': (
                                f"a\n"
                                f"شماره ملی: {result['National_ID']}\n"
                                f"نام: {result['First Name']}\n"
                                f"نام خانوادگی: {result['Last Name']}\n"
                                f"تاریخ تولد: {persian_birth_date}\n"
                                f"نام پدر: {result['Father Name']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}

        elif type == 1:
            text_output = {'passage': (f"b\n"
                                       f"{result['National ID Serial']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 2:
            text_output = {'passage': (
                            f"c"),
                                # f"نام: {result['First Name']}\n"
                                # f"نام خانوادگی: {result['Last Name']}\n"
                                # f"نام پدر: {result['Father Name']}\n"
                                # f"تاریخ تولد: {persian_birth_date}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 3:
            text_output = {'passage': (
                f"d"),
                # f"نام: {result['First Name']}\n"
                # f"نام خانوادگی: {result['Last Name']}\n"
                # f"نام پدر: {result['Father Name']}\n"
                # f"تاریخ تولد: {persian_birth_date}"),
                'class': result['Class'],
                'person_coordinate': result['Person_coordinate'],
                'rotation': result['Rotation'],
                'scale': result['Scale'],
                'transport': result['Transport']}

        return text_output

    def __getitem__(self, item):

        # Use the valid index
        MAX_LENGTH = 160
        EOS_token = 1
        file_name = self.all_data.iloc[item]['location'].split('/')[-1]
        labels = pd.read_csv(self.all_data.iloc[item]['csv_file'], encoding='UTF-8-SIG', converters={'feature': literal_eval})
        result = self.search_by_national_id(int(file_name.split('.')[0].split('_')[0]), int(file_name.split('.')[0].split('_')[1]), labels)
        target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127
        if result:
            try:
                image = cv2.imread(self.all_data.iloc[item]['location'])
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids
            except Exception as e:
                with open("/home/kasra/PycharmProjects/Larkimas/hs_err_pid11016.log", "a") as log_file:
                    log_file.write(f"An error occurred while reading {file_name}: {e}\n")
                image = cv2.imread(self.all_data.iloc[item]['location'])
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids

        if result:
            if self.transform is not None:
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.all_data)



class just_ID_card_DataLoader(Dataset):
    """
    Args:
        image_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, )
    """
    def __init__(self, data_file: str, transform=None):
        super(All_ID_card_DataLoader, self).__init__()

        self.data_file = data_file
        self.all_data = pd.read_csv(self.data_file)
        self.transform = transform

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
            a = str(int(row['NATIONAL_ID']))
            while len(list(a)) != 10:
                a = '0' + a

            if row['SCALE'] == 0:
                scale = 1
            else:
                scale = row['SCALE']

            result = {
                'National_ID': a,
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
                'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
                'Class': int(row['CLASS']),
                'Person_coordinate': np.round(np.array(literal_eval(row['PERSON_COORD'])).reshape(8)/720, 2),
                'Rotation': (row['ROTATION'] + 1)/2,
                'Scale': scale,
                'Transport': np.round(((np.array(literal_eval(row['TRANSPORT'])).reshape(2)/360) +
                             np.ones_like(np.array(literal_eval(row['TRANSPORT'])).reshape(2)))/2, 2),
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
            text_output = {'passage': (
                                f"a\n"
                                f"شماره ملی: {result['National_ID']}\n"
                                f"نام: {result['First Name']}\n"
                                f"نام خانوادگی: {result['Last Name']}\n"
                                f"تاریخ تولد: {persian_birth_date}\n"
                                f"نام پدر: {result['Father Name']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}

        elif type == 1:
            text_output = {'passage': (f"b\n"
                                       f"{result['National ID Serial']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 2:
            text_output = {'passage': (
                            f"c"),
                                # f"نام: {result['First Name']}\n"
                                # f"نام خانوادگی: {result['Last Name']}\n"
                                # f"نام پدر: {result['Father Name']}\n"
                                # f"تاریخ تولد: {persian_birth_date}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 3:
            text_output = {'passage': (
                f"d"),
                # f"نام: {result['First Name']}\n"
                # f"نام خانوادگی: {result['Last Name']}\n"
                # f"نام پدر: {result['Father Name']}\n"
                # f"تاریخ تولد: {persian_birth_date}"),
                'class': result['Class'],
                'person_coordinate': result['Person_coordinate'],
                'rotation': result['Rotation'],
                'scale': result['Scale'],
                'transport': result['Transport']}

        return text_output

    def __getitem__(self, item):

        # Use the valid index
        MAX_LENGTH = 160
        EOS_token = 1
        file_name = self.all_data.iloc[item]['location'].split('/')[-1]
        labels = pd.read_csv(self.all_data.iloc[item]['csv_file'], encoding='UTF-8-SIG', converters={'feature': literal_eval})
        result = self.search_by_national_id(int(file_name.split('.')[0].split('_')[0]), int(file_name.split('.')[0].split('_')[1]), labels)
        target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127
        if result:
            try:
                image = cv2.imread(self.all_data.iloc[item]['location'])
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids
            except Exception as e:
                with open("/home/kasra/PycharmProjects/Larkimas/hs_err_pid11016.log", "a") as log_file:
                    log_file.write(f"An error occurred while reading {file_name}: {e}\n")
                image = cv2.imread(self.all_data.iloc[item]['location'])
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids

        if result:
            if self.transform is not None:
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.all_data)

class All_ID_card_DataLoader(Dataset):
    """
    Args:
        image_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, )
    """
    def __init__(self, data_file: str, transform=None):
        super(All_ID_card_DataLoader, self).__init__()

        self.data_file = data_file
        self.all_data = pd.read_csv(self.data_file)
        self.transform = transform

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
            a = str(int(row['NATIONAL_ID']))
            while len(list(a)) != 10:
                a = '0' + a

            if row['SCALE'] == 0:
                scale = 1
            else:
                scale = row['SCALE']

            result = {
                'National_ID': a,
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
                'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
                'Class': int(row['CLASS']),
                'Person_coordinate': np.round(np.array(literal_eval(row['PERSON_COORD'])).reshape(8)/720, 2),
                'Rotation': (row['ROTATION'] + 1)/2,
                'Scale': scale,
                'Transport': np.round(((np.array(literal_eval(row['TRANSPORT'])).reshape(2)/360) +
                             np.ones_like(np.array(literal_eval(row['TRANSPORT'])).reshape(2)))/2, 2),
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
            text_output = {'passage': (
                                f"a\n"
                                f"شماره ملی: {result['National_ID']}\n"
                                f"نام: {result['First Name']}\n"
                                f"نام خانوادگی: {result['Last Name']}\n"
                                f"تاریخ تولد: {persian_birth_date}\n"
                                f"نام پدر: {result['Father Name']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}

        elif type == 1:
            text_output = {'passage': (f"b\n"
                                       f"{result['National ID Serial']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 2:
            text_output = {'passage': (
                            f"c"),
                                # f"نام: {result['First Name']}\n"
                                # f"نام خانوادگی: {result['Last Name']}\n"
                                # f"نام پدر: {result['Father Name']}\n"
                                # f"تاریخ تولد: {persian_birth_date}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 3:
            text_output = {'passage': (
                f"d"),
                # f"نام: {result['First Name']}\n"
                # f"نام خانوادگی: {result['Last Name']}\n"
                # f"نام پدر: {result['Father Name']}\n"
                # f"تاریخ تولد: {persian_birth_date}"),
                'class': result['Class'],
                'person_coordinate': result['Person_coordinate'],
                'rotation': result['Rotation'],
                'scale': result['Scale'],
                'transport': result['Transport']}

        return text_output

    def __getitem__(self, item):

        # Use the valid index
        MAX_LENGTH = 160
        EOS_token = 1
        file_name = self.all_data.iloc[item]['location'].split('/')[-1]
        labels = pd.read_csv(self.all_data.iloc[item]['csv_file'], encoding='UTF-8-SIG', converters={'feature': literal_eval})
        result = self.search_by_national_id(int(file_name.split('.')[0].split('_')[0]), int(file_name.split('.')[0].split('_')[1]), labels)
        target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127
        if result:
            try:
                image = cv2.imread(self.all_data.iloc[item]['location'])
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids
            except Exception as e:
                with open("/home/kasra/PycharmProjects/Larkimas/hs_err_pid11016.log", "a") as log_file:
                    log_file.write(f"An error occurred while reading {file_name}: {e}\n")
                image = cv2.imread(self.all_data.iloc[item]['location'])
                label = self.format_result_to_persian_text(result, result['Class'])
                encoded = [char2ind(char) for char in list(label['passage'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_passage'] = target_ids

        if result:
            if self.transform is not None:
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.all_data)



class cropped_DataLoader(Dataset):
    """
    Args:
        image_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, )
    """
    def __init__(self, data_file: str, transform=None):
        super(cropped_DataLoader, self).__init__()

        self.data_file = data_file
        self.all_data = pd.read_csv(self.data_file)
        self.transform = transform

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
            a = str(int(row['NATIONAL_ID']))
            while len(list(a)) != 10:
                a = '0' + a

            if row['SCALE'] == 0:
                scale = 1
            else:
                scale = row['SCALE']

            result = {
                'National_ID': a,
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']).replace('/', '*'),
                'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
                'Class': int(row['CLASS']),
                'Person_coordinate': np.round(np.array(literal_eval(row['PERSON_COORD'])).reshape(8)/720, 2),
                'Rotation': (row['ROTATION'] + 1)/2,
                'Scale': scale,
                'Transport': np.round(((np.array(literal_eval(row['TRANSPORT'])).reshape(2)/360) +
                             np.ones_like(np.array(literal_eval(row['TRANSPORT'])).reshape(2)))/2, 2),
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
        result['Persian Birth Date'] = self.convert_to_persian_numbers(result['Persian Birth Date'])

        # Format the text output
        if type == 0:
            text_output = {'passage': (
                                f"a\n"
                                f"شماره ملی: {result['National_ID']}\n"
                                f"نام: {result['First Name']}\n"
                                f"نام خانوادگی: {result['Last Name']}\n"
                                f"تاریخ تولد: {persian_birth_date}\n"
                                f"نام پدر: {result['Father Name']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}

        elif type == 1:
            text_output = {'passage': (f"b\n"
                                       f"{result['National ID Serial']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 2:
            text_output = {'passage': (
                            f"c"),
                                # f"نام: {result['First Name']}\n"
                                # f"نام خانوادگی: {result['Last Name']}\n"
                                # f"نام پدر: {result['Father Name']}\n"
                                # f"تاریخ تولد: {persian_birth_date}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 3:
            text_output = {'passage': (
                f"d"),
                # f"نام: {result['First Name']}\n"
                # f"نام خانوادگی: {result['Last Name']}\n"
                # f"نام پدر: {result['Father Name']}\n"
                # f"تاریخ تولد: {persian_birth_date}"),
                'class': result['Class'],
                'person_coordinate': result['Person_coordinate'],
                'rotation': result['Rotation'],
                'scale': result['Scale'],
                'transport': result['Transport']}

        return result

    def __getitem__(self, item):

        # Use the valid index
        MAX_LENGTH = 25
        EOS_token = 1
        file_name = self.all_data.iloc[item]['location'].split('/')[-1]
        labels = pd.read_csv(self.all_data.iloc[item]['csv_file'], encoding='UTF-8-SIG', converters={'feature': literal_eval})
        result = self.search_by_national_id(int(file_name.split('.')[0].split('_')[0]), int(file_name.split('.')[0].split('_')[1]), labels)
        target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127
        if result:
            try:
                image = self.all_data.iloc[item]['location']
                label = self.format_result_to_persian_text(result, result['Class'])

                encoded = [char2ind(char) for char in list(label['National_ID'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_National_ID'] = target_ids
                target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127

                encoded = [char2ind(char) for char in list(label['First Name'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_First Name'] = target_ids
                target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127

                encoded = [char2ind(char) for char in list(label['Last Name'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_Last Name'] = target_ids
                target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127

                encoded = [char2ind(char) for char in list(label['Father Name'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_Father Name'] = target_ids
                target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127

                encoded = [char2ind(char) for char in list(label['Persian Birth Date'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_Persian Birth Date'] = target_ids
                target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127

                encoded = [char2ind(char) for char in list(label['National ID Serial'])]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label['encoded_National ID Serial'] = target_ids
                target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127
            except Exception as e:
                print(f"An error occurred while reading : {e}\n")

        # if result:
        #     if self.transform is not None:
        #         image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.all_data)

# result['National_ID']
# result['First Name']
# result['Last Name']
# result['Father Name']
# result['Persian Birth Date']
# result['National ID Serial']




class final_dataLoader(Dataset):
    """
    Args:
        image_folder : str

    Return:
        image : np.array : (N, H, W, C)
        labels  : np.array : (N, )
    """
    def __init__(self, data_file: str, transform=None):
        super(final_dataLoader, self).__init__()

        self.file = data_file
        self.transform = transform
        self.data_files = os.listdir((self.file))
        self.label_types= ['First Name',
                          'Last Name',
                          'Father Name',
                          'Persian Birth Date',
                          'National_ID',
                          'National ID Serial']

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
            a = str(int(row['NATIONAL_ID']))
            while len(list(a)) != 10:
                a = '0' + a

            if row['SCALE'] == 0:
                scale = 1
            else:
                scale = row['SCALE']

            result = {
                'National_ID': a,
                'First Name': row['FIRST_NAME'],
                'Last Name': row['LAST_NAME'],
                'Father Name': row['FATHER_NAME'],
                'Birth Date': row['BIRTH_DATE'],
                'Persian Birth Date': self.convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
                'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
                'Class': int(row['CLASS']),
                'Person_coordinate': np.round(np.array(literal_eval(row['PERSON_COORD'])).reshape(8)/720, 2),
                'Rotation': (row['ROTATION'] + 1)/2,
                'Scale': scale,
                'Transport': np.round(((np.array(literal_eval(row['TRANSPORT'])).reshape(2)/360) +
                             np.ones_like(np.array(literal_eval(row['TRANSPORT'])).reshape(2)))/2, 2),
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
        result['Persian Birth Date'] = self.convert_to_persian_numbers(result['Persian Birth Date'])

        # Format the text output
        if type == 0:
            text_output = {'passage': (
                                f"a\n"
                                f"شماره ملی: {result['National_ID']}\n"
                                f"نام: {result['First Name']}\n"
                                f"نام خانوادگی: {result['Last Name']}\n"
                                f"تاریخ تولد: {persian_birth_date}\n"
                                f"نام پدر: {result['Father Name']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}

        elif type == 1:
            text_output = {'passage': (f"b\n"
                                       f"{result['National ID Serial']}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 2:
            text_output = {'passage': (
                            f"c"),
                                # f"نام: {result['First Name']}\n"
                                # f"نام خانوادگی: {result['Last Name']}\n"
                                # f"نام پدر: {result['Father Name']}\n"
                                # f"تاریخ تولد: {persian_birth_date}"),
                           'class': result['Class'],
                           'person_coordinate': result['Person_coordinate'],
                           'rotation': result['Rotation'],
                           'scale': result['Scale'],
                           'transport': result['Transport']}
        elif type == 3:
            text_output = {'passage': (
                f"d"),
                # f"نام: {result['First Name']}\n"
                # f"نام خانوادگی: {result['Last Name']}\n"
                # f"نام پدر: {result['Father Name']}\n"
                # f"تاریخ تولد: {persian_birth_date}"),
                'class': result['Class'],
                'person_coordinate': result['Person_coordinate'],
                'rotation': result['Rotation'],
                'scale': result['Scale'],
                'transport': result['Transport']}

        return result

    def __getitem__(self, item):

        # Use the valid index
        MAX_LENGTH = 35
        EOS_token = 1
        file_name = self.data_files[item]
        label = file_name.split(',')[-1]
        character_label = label
        label_type = file_name.split(',')[-2]
        target_ids = np.ones((MAX_LENGTH), dtype=np.int32) * 127
        image_path = self.file +'/'+ file_name
        typee = np.zeros((len(self.label_types)))
        typee[self.label_types.index(label_type)] = 1
        typee = torch.tensor(typee).float()

        try:
            if label_type == 'Persian Birth Date':
                character_label = label.replace('*', '/')
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = image.reshape(image.shape[0], image.shape[1], 1)
                encoded = [char2ind(char) for char in label.replace('*', '/')]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label = target_ids

            elif label_type == 'National_ID':
                character_label = self.convert_to_persian_numbers(label)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = image.reshape(image.shape[0], image.shape[1], 1)
                encoded = [char2ind(char) for char in self.convert_to_persian_numbers(label)]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label = target_ids

            else:
                character_label = label
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = image.reshape(image.shape[0], image.shape[1], 1)
                encoded = [char2ind(char) for char in label]
                encoded.append(EOS_token)
                target_ids[:len(encoded)] = encoded
                label = target_ids



        except Exception as e:
            print(f"An error occurred while reading : {e}\n")


        if self.transform is not None:
            image = self.transform(image)

        return image, label, typee, character_label

    def __len__(self):
        return len(self.data_files)

# result['National_ID']
# result['First Name']
# result['Last Name']
# result['Father Name']
# result['Persian Birth Date']
# result['National ID Serial']