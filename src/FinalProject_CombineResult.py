# combine different results
# import keras
import pandas as pd
import sys
import os

output_path = sys.argv[1]

result_paths = [
    'class0_12.csv',
    'class13.csv'
]
result_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

input_csv = []
columns = None
output_dict = {}
for idx, path in enumerate(result_paths):
    if not os.path.isfile(path):
        print('not found:', path)
    else:
        df = pd.read_csv(path, dtype='O')
        if idx == 0:
            columns = df.columns.values
            print(columns)
            output_dict[columns[0]] = df[columns[0]]
        input_csv.append(df)

# print(output_dict)
for col in range(14):
    output_dict[columns[col + 1]] = input_csv[result_label[col]][columns[col + 1]]

pd.DataFrame(output_dict).to_csv(output_path, index=False)





