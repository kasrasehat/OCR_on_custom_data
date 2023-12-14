import pandas as pd
import os

def create_csv_if_not_exists(csv_path):
    if not os.path.exists(csv_path):
        # Create an empty DataFrame with the same columns as BB
        df = pd.DataFrame(columns=['location','csv_file'])

        # Save the empty DataFrame to a new CSV file named "AAA.csv"
        df.to_csv(csv_path, encoding='UTF-8-SIG', index=False)
        print(f"csv file '{csv_path}' created.")
    else:
        print(f"csv file '{csv_path}' already exists.")

data = pd.read_csv('/home/kasra/kasra_files/data-shenasname/data_loc')
csv_path = '/home/kasra/kasra_files/data-shenasname/just_id_card'
create_csv_if_not_exists(csv_path)
new_csv = pd.read_csv(csv_path, encoding='UTF-8-SIG')
p = 0

for i in range(len(data)):
    if data.iloc[i]['location'].split('.')[0].split('_')[-1] != '2':
        new_csv.loc[p, :] = data.loc[i, :]
        p = p + 1

new_csv.to_csv(csv_path, encoding='UTF-8-SIG', index=False)