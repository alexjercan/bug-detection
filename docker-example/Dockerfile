FROM python:3.8

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY download_models.py download_models.py

RUN python download_models.py

COPY . .

CMD ["streamlit", "run", "app.py"]

