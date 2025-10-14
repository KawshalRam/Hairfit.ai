import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 6]

# Data concatenation
path = r'C:\Users\kawsh\coursera-test\OneDrive\Desktop\Manipal\PROJECTS\HairFit.ai\hairfit-frontend\Backend\amazon'
all_files = [x for x in os.listdir(path) if x.endswith(".csv")]

li = []
for filename in all_files:
    df = pd.read_csv(os.path.join(path, filename), index_col=None, header=0)

    parts = filename.replace('.csv', '').split('_')
    df["Brand"] = parts[0] if len(parts) > 0 else None
    df["Line"] = parts[1] if len(parts) > 1 else None

    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df = df[df["Rating"].str.contains("stars", na=False)]
df["RatingNumeric"] = df["Rating"].str.split().str[0].astype(float)
df["Title"] = df["Title"].str.lower()
df["Review"] = df["Review"].str.lower()

df.to_csv('haircare_data.csv', index=False)
print("✅ Dataset saved successfully as haircare_data.csv")
