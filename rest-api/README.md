# Bug Detection Container

Docker image using flask to test the bug detection pipeline.

## Quickstart

### Docker Compose

```console
docker-compose up
```

### Docker

```console
docker build . --tag bug-repair-flask
docker run --it -p 8080:80 bug-repair-flask:latest
```

This will download all the dependencies from `requirements.txt` and the models
from huggingface. Then it will run the Flask rest api on the address:
`localhost:8080`.

## Rest API

The application responds to the following routes:

- `POST /api/inference` The application will read the field "source\_code" from
  the body of the request and will respond with a json object containing three
  fields "error\_description", "token\_class" and "source\_code". The error
  description will contain the description of the error in natural language,
  the token class will contain an array for each character of the source code
  with 1 for buggy and 0 for correct, and the source code field will contain
  the repaired source code.
