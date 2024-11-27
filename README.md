# WRO 2022 Future Engineers Randomization app

## Run from CLI

- `sudo apt-get update`
- `sudo apt-get install -y libgl1-mesa-glx`
- `pip install -r requirements.txt`
- `gunicorn -w 2 app:app`

## Run by Docker

- `docker build -t fe-randomization-app .`
- `docker run -ti --rm -e HOST=0.0.0.0 -e PORT=8000 -p 8000:8000 fe-randomization-app`
