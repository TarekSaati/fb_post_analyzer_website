
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
import pandas as pd
import numpy as np
import json

def log_transform(x):
    print(x)
    return np.log(x + 1)

def get_fb_dataset(path: str):
    # Open the dataset meta-file as a dict to navigate across topics and pages
    with open('app/topics.json','r') as f:
        topics_pages = json.load(f)

    ds = pd.DataFrame()
    index_offset = 0

    # reading & concatenating data from csv files into a single dataframe
    for topic in topics_pages.keys():
        for pageName in topics_pages[topic]:
            dataset_path = f'{path}/{topic}/{pageName}/data.csv'
            df = pd.read_csv(dataset_path)
            # processing Nan values
            df.fillna(0, inplace=True)
            
            # add topic & pageName info as feature columns
            df['topic'] = [topic] * df.shape[0]
            df['pagename'] =[pageName] * df.shape[0]

            # Offsetting IDs for appended posts
            df['PostId'] = df['PostId'].astype('Int32')
            df['PostId'] = df['PostId'].apply(lambda x: x + index_offset)
            index_offset += df.shape[0]

            # renaming & setting datetimes
            df['time'] = pd.to_datetime(df['time'])
            df.rename(columns={'ranking': 'values'}, inplace=True)
            df['timestamp'] = df['timestamp'].astype('Float32')

            # reformatting timestamps to per-hour ascending format
            df['timestamp'] = 1+(df['timestamp'] - np.min(df['timestamp']))/3600
            
            # appending the current slice to the dataframe
            ds = pd.concat([ds, df])

            # setting the unique post ID as the index of dataframe
            ds.set_index('PostId', inplace=True)

    return ds

def process_dataset(ds: pd.DataFrame, test_ratio=.2, seed=1234):
    # label encoding using scikit-learn label encoder
    topic_enc = LabelEncoder()
    ds['topic'] = topic_enc.fit_transform(ds['topic'])

    # feature-label formatting
    X = ds.drop(['time', 'pagename', 'topic'], axis=1)
    y = ds['topic'].to_numpy(dtype=np.int32)

    # scaling & normalizing dataset features
    std_scaler = StandardScaler()
    transformer = FunctionTransformer(log_transform)
    X = transformer.fit_transform(X)
    X = std_scaler.fit_transform(X)
    
    return X, y
