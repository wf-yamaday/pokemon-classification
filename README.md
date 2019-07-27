# pokemon-classification

pokemon classification app with TensorFlow

## How to Run

### for macOS

- python 3.7

```
$ pip install -r requirements.txt
$ flask run
```

### Docker

```
$ docker image build -t pokemon:latest .
$ docker container run -d -p 5000:5000 pokemon:latest
```
