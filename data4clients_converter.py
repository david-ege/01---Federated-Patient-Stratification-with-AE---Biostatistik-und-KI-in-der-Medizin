import pandas as pd

df = pd.read_csv("./data.tsv", sep="\t",index_col=0)

df_final = df.T.apply(pd.to_numeric, errors='coerce')

metadata = pd.read_csv("./metadata.tsv", sep="\t")
df_final = df_final.sample(frac=1, random_state=30, ignore_index=True)
metadata = metadata.sample(frac=1, random_state=30, ignore_index=True)

splitsize = len(df_final)//3

df1=df_final.iloc[:splitsize]
df2=df_final.iloc[splitsize:2*splitsize]
df3=df_final.iloc[2*splitsize:]

metadata1 = metadata.iloc[:splitsize]
metadata2 = metadata.iloc[splitsize:2*splitsize]
metadata3 = metadata.iloc[2*splitsize:]

df_final.to_csv("./data/client1/allData.csv")
df_final.to_csv("./data/client2/allData.csv")
df_final.to_csv("./data/client3/allData.csv")
df1.to_csv("./data/client1/localData.csv", sep=";")
df2.to_csv("./data/client2/localData.csv", sep=";")
df3.to_csv("./data/client3/localData.csv", sep=";")

metadata1.to_csv("./data/client1/labelc1.csv", sep=";")
metadata2.to_csv("./data/client2/labelc2.csv", sep=";")
metadata3.to_csv("./data/client3/labelc3.csv", sep=";")
