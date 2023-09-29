from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import logging


# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def load_preprocessed_data(train_path, val_path):
    try:
        train_data = load_from_disk(train_path)
        val_data = load_from_disk(val_path)
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

import argparse
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--val_data_path', required=True)
    args = parser.parse_args()

    train_path = args.train_data_path
    val_path = args.val_data_path

    train_data, val_data = load_preprocessed_data(train_path, val_path)
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
