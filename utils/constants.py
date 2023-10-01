import os
from pathlib import Path

BUCKET_NAME = os.environ.get('SENTIMENT_BUCKET')

# These are preprocessed data paths and zips in S3 bucket. 
YELP_DATA_DIR_PATH = Path('my_datasets/yelp_dataset/')
TRAIN_DATA_ZIP_FILENAME = Path("train_set.zip") # zipped of preprocessed huggingFace data dump, which was done using save_to_disk() method.
VAL_DATA_ZIP_FILENAME = Path("val_set.zip")

# savuing model into s3
MODEL_SAVING_DIR_S3 = Path("trained_models")





