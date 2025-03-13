FROM odsai/df25-baseline:1.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ \
        curl \
        git \
        libomp-dev && \
    pip install --upgrade pip setuptools wheel && \
    pip install dill pymorphy3 nltk datasets transformers fasttext-wheel