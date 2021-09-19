FROM python:3.8.8

WORKDIR /code

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY templates ./templates

ENTRYPOINT [ "gunicorn", "-w", "2", "app:app" ]