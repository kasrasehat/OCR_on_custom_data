import pandas as pd
import chardet


# Detect the encoding of the file
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


# Your file path
file_path = 'E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/ai_metadata_20230528.csv'

# Detect encoding
detected_encoding = detect_encoding(file_path)

# Read the file using the detected encoding
df = pd.read_csv(file_path, encoding=detected_encoding)

# Write the DataFrame back to a new CSV file with UTF-8 encoding
new_file_path = 'E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/ai_metadata_20230528.csv'
df.to_csv(new_file_path, encoding='UTF-8-SIG', index=False)