import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import logging
import re
import boto3
import shutil
import os
from utils import constants

# Text cleaning function (customize as needed)
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '[URL]', text)
    # Remove Emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # Remove new line and line breaks
    text = text.replace('\n', ' ').replace('\r', '').strip()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Optional: Lowercase (Depends on whether your DistilBERT model is cased or not)
    # text = text.lower()

    return text

def load_dataset(dataset_path, chunk_size):
    try:
        json_reader = pd.read_json(dataset_path, lines=True, chunksize=chunk_size)
        print("Dataset Loaded!")
        return next(json_reader)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return None

def clean_data(df):
    try:
        df.drop(columns=['review_id', 'user_id', 'business_id'], inplace=True)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df['clean_text'] = df['text'].apply(clean_text)
    except Exception as e:
        logging.error(f"Data cleaning failed: {e}")

def feature_engineering(df):
    try:
        df['stars'] = df['stars'] - 1
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")

def preprocess_data(df):
    try:
        train_df, val_df = train_test_split(df, test_size=0.2)
        train_df['stars'] = train_df['stars'].astype(int)
        val_df['stars'] = val_df['stars'].astype(int)

        train_data = Dataset.from_pandas(train_df[['text', 'stars']].rename(columns={'text': 'text', 'stars': 'label'}))
        val_data = Dataset.from_pandas(val_df[['text', 'stars']].rename(columns={'text': 'text', 'stars': 'label'}))        

        return val_data, train_data
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return None, None

# Initialize S3 client
s3 = boto3.client('s3')
def dump_to_local_and_s3(train_data, val_data, train_dumpyard_path, val_dumpyard_path):
    try:

        # Using Hugging Face's Datasets
        train_data.save_to_disk(train_dumpyard_path) # "./datasets/train_data/"
        val_data.save_to_disk(val_dumpyard_path) # "./datasets/val_data/"
        print('Saved Training data: ',train_dumpyard_path)
        print('Saved Validation data: ',val_dumpyard_path)        

        # Zip the folders
        shutil.make_archive(local_train_path, 'zip', local_train_path)
        shutil.make_archive(local_val_path, 'zip', local_val_path)

        # Upload zipped folders to S3
        s3.upload_file(Filename=local_train_path + '.zip', Bucket=constants.BUCKET_NAME, Key=train_dumpyard_path + '.zip')
        s3.upload_file(Filename=local_val_path + '.zip', Bucket=constants.BUCKET_NAME, Key=val_dumpyard_path + '.zip')

        print(f"Saved Training data to S3: {train_dumpyard_path}.zip")
        print(f"Saved Validation data to S3: {val_dumpyard_path}.zip")
    except Exception as e:
        print(f"Error dumping to S3: {e}")


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    args = parser.parse_args()

    df = load_dataset(args.dataset_path, args.chunk_size)
    if df is not None:
        clean_data(df)
        print("Dataset Cleaned!")
        feature_engineering(df)
        print("Dataset Engineered!")
        train_data, val_data = preprocess_data(df)
        print("Dataset Cleaned!")

        # Saving data locally and s3
        train_dumpyard_path = '/'.join(args.dataset_path.split('/')[:-1]) + '/train_set/'
        val_dumpyard_path = '/'.join(args.dataset_path.split('/')[:-1]) + '/val_set/'
        dump_to_local_and_s3(train_data,val_data,train_dumpyard_path,val_dumpyard_path)



