import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

#'Average User Rating'
classname = 'Average User Rating'
max_size = df[classname].value_counts().max()
lst = [df]
for class_index, group in df.groupby(classname):
    lst.append(group.sample(max_size-len(group), replace=True))
frame_new = pd.concat(lst)

frame_new.to_csv(output_file, index=False)
