from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
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
    train_path = "path/to/directory_train"
    val_path = "path/to/directory_val"

    train_data, val_data = load_preprocessed_data(train_path, val_path)
    if train_data and val_data:
        train_data_tokenized, val_data_tokenized = tokenize_data(train_data, val_data, tokenize_function)  # Replace with your actual tokenize_function

        set_pytorch_format(train_data_tokenized)
        set_pytorch_format(val_data_tokenized)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs'
        )

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(train_data_tokenized['label'].unique()))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data_tokenized,
            eval_dataset=val_data_tokenized
        )

        trainer.train()
