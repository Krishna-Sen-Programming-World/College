from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

df = pd.read_csv(r"golf_dataset.csv")
print(df.head())

#Display the whole dataframe in tabular form
display(df)

#Display the description of the dataset
display(df.describe())

#%%

#Calculate shanon's entropy of the dataset
count_0 = count_1 = data_count = 0
for i in range(len(df.index)) :
    data_count = data_count + 1
    if(df.iat[i,4] == 'No') :
        count_0 = count_0 + 1
    else :
        count_1 = count_1 + 1

#print("NO Count: " + str(count_0))
#print("Yes Count: " + str(count_1))

prob_0 = count_0 / data_count
prob_1 = count_1 / data_count
initial_entropy = - ((prob_0 * math.log(prob_0, 2)) + (prob_1 * math.log(prob_1, 2)))

print("initial_entropy: " + str(initial_entropy))

#%%

#Let us create a dataframe to hold entropies of each feature
feature_entropy = np.empty(len(df.columns)-1)
entropy_gain = np.empty(len(df.columns)-1)

#Now, let us calculate the entropy for each column
unique_counts = df.nunique(0)
for i in range(len(df.columns)-1) :
    feature_entropy[i] = 0
    feature = df.columns[i]
    unique_count = unique_counts[i]
    unique_values = pd.unique(df[df.columns[i]])
    print(feature + ": " + str(unique_count))
    print(unique_values)
    for j in range(unique_count) :
        unique_value = unique_values[j]
        feature_count_0 = feature_count_1 = feature_count = 0
        for k in range(len(df.index)) :
            if(df.iat[k,i] == unique_value) :
                feature_count = feature_count + 1
                if(df.iloc[k]['Play Golf'] == "No") :
                    feature_count_0 = feature_count_0 + 1
                else :
                    feature_count_1 = feature_count_1 + 1

        prob_0 = feature_count_0 / feature_count
        prob_1 = feature_count_1 / feature_count
        if((prob_0 != 0) and (prob_1 != 0)) :
            local_entropy = - ((prob_0 * math.log(prob_0, 2)) + (prob_1 * math.log(prob_1, 2)))
        else :
            local_entropy = 0
        feature_entropy[i] = feature_entropy[i] + (feature_count / data_count) * local_entropy
        entropy_gain[i] = initial_entropy - feature_entropy[i]
    print("feature entropy: " + str(feature_entropy[i]))
    print("entropy gain: " + str(entropy_gain[i]))

print("\nFinal Features And Corresponding Gains\n---------------------------------")
for i in range(len(df.columns)-1) :
    print(df.columns[i] + ":\t" + str(entropy_gain[i]))
 