import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

mypath = 'C:\\Users\\DARIO-DELL\\Desktop\\Try\\'
analysis_filename = 'analysis_result.csv'            
classes = ['yes', 'no']


df = pd.read_csv(mypath+analysis_filename)
df = df.reindex(columns=['watch_sample','phone_sample','class','acc_similarity_2b','vel_similarity_2b','acc_similarity_3b','vel_similarity_3b'])

b1 = df.boxplot(by=['class'],column=['acc_similarity_2b'], grid=False)
b2 = df.boxplot(by=['class'],column=['vel_similarity_2b'], grid=False)
b3 = df.boxplot(by=['class'],column=['acc_similarity_3b'], grid=False)
b4 = df.boxplot(by=['class'],column=['vel_similarity_3b'], grid=False)


plt.figure()
for cl in classes:
    # Subset to the classes
    subset = df[df['class'] == cl]    
        
    # Draw the density plot
    sns.distplot(subset['vel_similarity_3b'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 label = cl)