FROM odsai/df25-baseline:1.0

RUN apt-get update && apt-get install -y \
	git \
    build-essential \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*
	
RUN pip install --upgrade pip setuptools wheel
RUN pip install dill pymorphy3 nltk datasets transformers faiss-gpu-cu12 sentence_transformers