from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import logging
import argparse
import yaml
import shutil
import boto3

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


s3 = boto3.client('s3')
def load_preprocessed_data(): #loading from s3
    try:

        # bring zip files from s3 to local dir and unzip
        train_path = str(YELP_DATA_DIR_PATH / TRAIN_DATA_ZIP_FILENAME)
        val_path = str(YELP_DATA_DIR_PATH / VAL_DATA_ZIP_FILENAME)
        s3.download_file(Bucket=constants.BUCKET_NAME, Key=train_path, Filename=train_path)
        s3.download_file(Bucket=constants.BUCKET_NAME, Key=val_path, Filename=val_path)

        print('downloaded from s3')
        #unzip
        shutil.unpack_archive(TRAIN_DATA_ZIP_FILENAME, YELP_DATA_DIR_PATH)
        shutil.unpack_archive(TRAIN_DATA_ZIP_FILENAME, YELP_DATA_DIR_PATH)
        print('Unzipped to ',YELP_DATA_DIR_PATH)

        
        #get unzip path as train_path
        train_data = load_from_disk(YELP_DATA_DIR_PATH)
        val_data = load_from_disk(YELP_DATA_DIR_PATH)
        return train_data, val_data
    except Exception as e:
        logging.error(f"Failed to load preprocessed data: {e}")
        return None, None

def tokenize_data(train_data, val_data, tokenize_function):
    try:
        print("Tokenizing Dataset....")
        train_data_tokenized = train_data.map(tokenize_function, batched=True)
        val_data_tokenized = val_data.map(tokenize_function, batched=True)
        return train_data_tokenized, val_data_tokenized
    except Exception as e:
        logging.error(f"Tokenization failed: {e}")
        return None, None

def set_pytorch_format(data):
    try:
        data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    except Exception as e:
        logging.error(f"Failed to set PyTorch format: {e}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_data_path', required=True)

    # parser.add_argument('--val_data_path', required=True)
    # parser.add_argument('--model_saving_dir', required=True)
    # args = parser.parse_args()

    # train_path = constants.BUCKET_NAME / YELP_DATA_DIR_PATH / TRAIN_DATA_ZIP_FILENAME
    # val_path = constants.BUCKET_NAME / YELP_DATA_DIR_PATH / TRAIN_DATA_ZIP_FILENAME
    # model_saving_dir = constants.BUCKET_NAME / constants.MODEL_SAVING_DIR_S3

    train_data, val_data = load_preprocessed_data()
    if train_data and val_data:
        train_data_tokenized, val_data_tokenized = tokenize_data(train_data, val_data, tokenize_function)  # Replace with your actual tokenize_function

        set_pytorch_format(train_data_tokenized)
        set_pytorch_format(val_data_tokenized)


        # Load the YAML config
        with open("training_config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)
        # Use the loaded config to set up TrainingArguments
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            per_device_eval_batch_size=config['per_device_eval_batch_size'],
            warmup_steps=config['warmup_steps'],
            weight_decay=config['weight_decay'],
            logging_dir=config['logging_dir']
        )

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(train_data_tokenized['label'].unique()))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data_tokenized,
            eval_dataset=val_data_tokenized
        )

        print("Training Model....")
        trainer.train()
        
        print("Saving trained Model....")
        #save to s3
        model.save_pretrained(model_saving_dir)