import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import logging

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

def dump_to_s3(train_data,val_data,train_dumpyard_path,val_dumpyard_path):
    # dump data into S3
    # Using Hugging Face's Datasets
    train_data.save_to_disk() # "./datasets/train_data/"
    print('Saved Training data: ',train_dumpyard_path)
    val_data.save_to_disk() # "./datasets/val_data/"
    print('Saved Validation data: ',val_dumpyard_path)


if __name__ == "__main__":
    dataset_path = 'your_dataset_path_here'
    chunk_size = 1000  # Replace with your actual chunk size

    df = load_dataset(dataset_path, chunk_size)
    if df is not None:
        clean_data(df)
        feature_engineering(df)
        train_data, val_data = preprocess_data(df)
        dump_to_s3(train_data,val_data,train_dumpyard_path,val_dumpyard_path)
