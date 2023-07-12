import os
import shutil
from new_old_classifier import detect_new_old

def copy_file(source_path, destination_path):
    shutil.copy2(source_path, destination_path)
    print("File copied successfully!")

detector = detect_new_old()
path = 'Data_source/all_data'
files = os.listdir(path)
file_paths = ['Data_source/all_data/front-ID', 'Data_source/all_data/beh-ID','Data_source/all_data/new-shen','Data_source/all_data/old-shen']
for file_path in file_paths:
    if os.path.isdir(file_path):
        print("File exists.")
    else:
        # Create the folder
        os.makedirs(file_path, exist_ok=False)
        print("Folder created.")

corrupted = []
for file in files:
    try:
        if int(file[-5]) == 0:
            destination_path = 'Data_source/all_data/front-ID'
        elif int(file[-5]) == 1:
            destination_path = 'Data_source/all_data/beh-ID'
        elif int(file[-5]) == 2:
            file1 = 'E:/codes_py/Larkimas/Data_source/all_data/' + file
            if detector.detect(file1) == 0:
                destination_path = 'Data_source/all_data/new-shen'
            elif detector.detect(file1) == 1:
                destination_path = 'Data_source/all_data/old-shen'


        source_path = 'Data_source/all_data/' + file


        # Call the function to copy the file
        copy_file(source_path, destination_path)

    except:
        corrupted.append(file)
        continue
