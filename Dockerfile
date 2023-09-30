# Use Hugging Face's transformer image as a base
FROM huggingface/transformers-pytorch-cpu

# Set the working directory in the container
WORKDIR /app-end-end-ml

# Copy your requirements file into the container
COPY requirements.txt /app-end-end-ml/

# Install additional Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system packages if needed
RUN apt-get update && apt-get install -y \
    zip \
    unzip

# Copy your scripts into the container
COPY datasets /app-end-end-ml/datasets
COPY preprocessing /app-end-end-ml/preprocessing
COPY utils /app-end-end-ml/utils
COPY _end_end_ml_pipeline.ipynb /app-end-end-ml/_end_end_ml_pipeline.ipynb

COPY train.py /app-end-end-ml/train.py
COPY training_config.yaml /app-end-end-ml/training_config.yaml

# Command to run when the container starts
CMD [ "bash" ]


# docker build -t end-end-ml-image . && docker stop end-end-container-name && docker rm end-end-container-name && docker run -it  --name end-end-container-name -d end-end-ml-image
