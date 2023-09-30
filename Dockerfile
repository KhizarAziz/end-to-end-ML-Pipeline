# Use Hugging Face's transformer image as a base
FROM huggingface/transformers-pytorch-cpu

# Set the working directory in the container
WORKDIR /endtoendml

# Copy your requirements file into the container
COPY requirements.txt /endtoendml/

# Install additional Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system packages if needed
RUN apt-get update && apt-get install -y \
    zip \
    unzip

# Copy your scripts into the container
COPY my_datasets /endtoendml/my_datasets
COPY preprocessing /endtoendml/preprocessing
COPY utils /endtoendml/utils
COPY _end_end_ml_pipeline.ipynb /endtoendml/_end_end_ml_pipeline.ipynb

COPY __init__.py /endtoendml/__init__.py
COPY train.py /endtoendml/train.py
COPY training_config.yaml /endtoendml/training_config.yaml

# Command to run when the container starts
CMD [ "bash" ]
