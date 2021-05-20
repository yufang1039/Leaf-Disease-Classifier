import pandas as pd

# read official csv file
train_origin = pd.read_csv('train.csv')

# initialize the dataframe
train_df = pd.DataFrame(columns=["image", "healthy", "scab", "frog_eye_leaf_spot", "rust", 
                                "complex", "powdery_mildew"])
train_df['image'] = train_origin['image']
train_df['healthy'] = [0] * len(train_origin)
train_df['scab'] = [0] * len(train_origin)
train_df['frog_eye_leaf_spot'] = [0] * len(train_origin)
train_df['rust'] = [0] * len(train_origin)
train_df['complex'] = [0] * len(train_origin)
train_df['powdery_mildew'] = [0] * len(train_origin)

labels = train_origin['labels']
print(labels[0])
for index in range(len(train_origin.index)):
    if "healthy" in labels[index]:
        train_df['healthy'][index] = 1
    else:
        if "scab" in labels[index]:
            train_df.at[index, 'scab'] = 1
        if "frog_eye_leaf_spot" in labels[index]:
            train_df.at[index, 'frog_eye_leaf_spot'] = 1
        if "rust" in labels[index]:
            train_df.at[index, 'rust'] = 1
        if "complex" in labels[index]:
            train_df.at[index, 'complex'] = 1
        if "powdery_mildew" in labels[index]:
            train_df.at[index, 'powdery_mildew'] = 1

train_df.to_csv("train_df.csv", index=False)