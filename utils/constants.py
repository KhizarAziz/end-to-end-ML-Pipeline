import os
BUCKET_NAME = os.environ.get('SENTIMENT_BUCKET')
DATA_PATH_IN_BUCKET = 'datasets/yelp_processed_data/'
TRAIN_DATA_ZIP_FILENAME = "train_set.zip"
VAL_DATA_ZIP_FILENAME = "val_set.zip"