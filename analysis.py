import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

mypath = 'C:\\Users\\DARIO-DELL\\Desktop\\Try\\'
result_filename = 'similarity_result.csv'
string_end_watch = '_watch_sample.csv'
string_end_phone = '_smartphone_sample.csv'

phone_file_names = list()
watch_file_names = list()
first_column = list()
second_column = list()
should_match = list()

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in files:
    if (f.endswith(string_end_phone)):
        phone_file_names.append(f)
    if (f.endswith(string_end_watch)):
        watch_file_names.append(f)

index = np.arange(len(watch_file_names)*len(phone_file_names))

for f1 in watch_file_names:
    print(f1)
    for f2 in phone_file_names:
        print(f1)
        first_column.append(f1)
        second_column.append(f2)
        if (f1[1:13] == f2[1:13]):
            should_match.append('yes')
        else:
            should_match.append('no')
            
df = pd.read_csv(mypath+result_filename)
df['watch_file_names'] = watch_file_names
