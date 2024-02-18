import pandas as pd
filename="SimLex-999.txt"

read_file=pd.read_csv(filename,delimiter='\t')

read_file=read_file.dropna()

req_column=['word1','word2','SimLex999']
subset_file=read_file[req_column]

subset_file.to_csv("taska_simlex_cleaned.csv",index=False)