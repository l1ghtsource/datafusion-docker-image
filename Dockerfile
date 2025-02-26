# Используем базовый образ
FROM odsai/df25-baseline:1.0

# Устанавливаем необходимые библиотеки
RUN pip install datasets transformers