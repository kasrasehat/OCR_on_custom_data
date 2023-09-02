import zipfile
import tarfile
import os
import time


def unzip_tar_gz_files_in_folder(folder_path):
    """Unzip .tar.gz files in a given folder into folders with the same name as the file."""
    files_in_folder = os.listdir(folder_path)
    total_files = len([f for f in files_in_folder if f.endswith('.tar.gz')])
    processed_files = 0

    for file_name in files_in_folder:
        if file_name.endswith('.tar.gz'):
            start_time = time.time()
            full_file_path = os.path.join(folder_path, file_name)
            extraction_folder_name = file_name.replace('.tar.gz', '')
            extraction_folder_path = os.path.join(folder_path, extraction_folder_name)

            if not os.path.exists(extraction_folder_path):
                os.makedirs(extraction_folder_path)

            with tarfile.open(full_file_path, 'r:gz') as file:
                try:
                    file.extractall(path=extraction_folder_path)
                except EOFError:
                    print(f"Skipping {file_name} due to EOFError.")

            end_time = time.time()
            processed_files += 1
            elapsed_time = end_time - start_time
            remaining_files = total_files - processed_files
            estimated_remaining_time = remaining_files * elapsed_time

            print(
                f"Unzipped {file_name} in {elapsed_time:.2f} seconds. Estimated time for remaining files: {estimated_remaining_time:.2f} seconds.")


unzip_tar_gz_files_in_folder('/home/kasra/kasra_files/data-shenasname')

