import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# close possible previously created plots
plt.close('all')

# path to folder containing the imported samples
mypath = 'C:\\Users\\DARIO-DELL\\Desktop\\Try\\'
# file containing the result of postprocessing.py script
analysis_filename = 'analysis_result.csv'
# list for detecting samples who should/should not match
classes = ['yes', 'no']

# FILE READING
df = pd.read_csv(mypath+analysis_filename)

fig, axes = plt.subplots(nrows=2, ncols=2) #used for 2D drawings

df.boxplot(ax=axes[0,0], by=['class'], column=['acc_similarity_2b'], grid=False)
df.boxplot(ax=axes[0,1], by=['class'], column=['acc_similarity_3b'], grid=False)
df.boxplot(ax=axes[1,0], by=['class'], column=['vel_similarity_2b'], grid=False)
df.boxplot(ax=axes[1,1], by=['class'], column=['vel_similarity_3b'], grid=False)

fig.suptitle('')

axes[0,0].set_xlabel("Class")
axes[0,0].set_ylabel("Similarity")
axes[0,1].set_xlabel("Class")
axes[0,1].set_ylabel("Similarity")
axes[1,0].set_xlabel("Class")
axes[1,0].set_ylabel("Similarity")
axes[1,1].set_xlabel("Class")
axes[1,1].set_ylabel("Similarity")


plt.tight_layout()
fig2 = plt.figure()

ax1 = fig2.add_subplot(221)
ax2 = fig2.add_subplot(222)
ax3 = fig2.add_subplot(223)
ax4 = fig2.add_subplot(224)

for cl in classes:
    # dataframe divided by classes
    subset = df[df['class'] == cl]    
        
    # Draw the density plots
    sns.distplot(subset['acc_similarity_2b'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 label = cl, ax=ax1)
    
    sns.distplot(subset['acc_similarity_3b'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 label = cl, ax=ax2)
    
    sns.distplot(subset['vel_similarity_2b'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 label = cl, ax=ax3)
    
    sns.distplot(subset['vel_similarity_3b'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 label = cl, ax=ax4)

plt.tight_layout()