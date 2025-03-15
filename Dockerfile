FROM python:3.11-slim

WORKDIR /lazy_ml_app

# To debug lightgbm related errors
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /lazy_ml_app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "lazyml.py"]
