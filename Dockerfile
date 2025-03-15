FROM python:3.11-slim

WORKDIR /lazy_ml_app

COPY . /lazy_ml_app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "lazyml.py"]
