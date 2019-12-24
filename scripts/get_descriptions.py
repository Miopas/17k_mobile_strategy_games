import pandas as pd
import sys

input_file = sys.argv[1]
ori_file = sys.argv[2]
out_file = sys.argv[3]

df = pd.read_csv(input_file)
labels = df['Average User Rating'].tolist()
IDs = df['ID'].tolist()

df_ori = pd.read_csv(ori_file)
output = {'Description':[], 'Average User Rating':[]}
for i, row in df_ori.iterrows():

    if (row['User Rating Count'] < 200):
        continue

    if row['ID'] in IDs:
        idx = IDs.index(row['ID'])
        #output['ID'].append(row['ID'])
        output['Average User Rating'].append(labels[idx])
        output['Description'].append(row['Description'])

pd.DataFrame(output).to_csv(out_file, sep='\t', index=False)

