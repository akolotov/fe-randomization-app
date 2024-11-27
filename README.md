# WRO 2024 Future Engineers Randomization app

The web app simplifies the randomization process by preparing a picture with random layouts for both Open and Obstacle challenges:

- For the Open Challenge
  - the inner walls configuration
  - the starting zone

- For the Obstacle Challenge
  - the obstacles positions
  - the starting zone
  - the parking lot section

Examples of the pictures:

| Open Challenge | Obstacle Challenge |
|:----:|:----:|
| ![image](https://github.com/user-attachments/assets/eab032fb-20b3-4eff-9d0a-32404de0ced8) | ![image](https://github.com/user-attachments/assets/937f0b5e-c089-4d7b-8c17-16c25cee9abc) |

## Run from CLI

- `sudo apt-get update`
- `sudo apt-get install -y libgl1-mesa-glx`
- `pip install -r requirements.txt`
- `gunicorn -w 2 app:app`

## Run by Docker

- `docker build -t fe-randomization-app .`
- `docker run -ti --rm -e HOST=0.0.0.0 -e PORT=8000 -p 8000:8000 fe-randomization-app`
