import pandas as pd
from sklearn.model_selection import train_test_split
import sys

input_file = sys.argv[1]

df = pd.read_csv(input_file)
train, test = train_test_split(df, test_size=0.2)
train.to_csv('train.csv', index=False);
test.to_csv('test.csv', index=False);
