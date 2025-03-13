FROM odsai/df25-baseline:1.0

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*
	
RUN apt install python3 python3-pip
	
RUN pip3 install dill pymorphy3 nltk datasets transformers fasttext