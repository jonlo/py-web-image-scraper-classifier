# py-web-image-scrapper-classifier

## Description

This application, exposes an API to scrap and classify images from a given URL.

It's fully dockerized so you can deploy anywhere.

You can easily select the tensorflow model to use for the classification by changing the `model_name` variable in the imageclassificator.py file.

The options are:

/scrap
To scrap the images from the given URL

/classify
To scrap and classify the images from the given URL

/image
To get the image from the given URL

Uses fastapi, tensorflow and uvicorn.

## Usage
If you have CUDA installed, you can just run uvicorn

```bash
uvicorn main:app --reload
```
go to http://localhost:8000/docs

### Docker

If you don't have CUDA installed, you can use the docker image

```bash
docker build -t py-web-image-scrapper-classifier .
docker run -p 8000:8000 py-web-image-scrapper-classifier
```
