# Используем базовый образ
FROM odsai/df25-baseline:1.0

# Устанавливаем необходимые библиотеки
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
    pip install "lightgbm==4.5.0" datasets transformers