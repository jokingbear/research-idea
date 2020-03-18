import pandas as pd


df = pd.read_csv("train.csv")
df["Pneumothorax"].value_counts()
