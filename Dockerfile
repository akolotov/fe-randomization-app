FROM python:3.8.8

WORKDIR /code

RUN apt-get update && \
    apt install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY templates ./templates

ENTRYPOINT [ "gunicorn", "app:app" ]
