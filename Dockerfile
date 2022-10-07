FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa 
# Install py39 from deadsnakes repository
RUN apt-get install python3.9 -y
# Install pip from standard ubuntu packages
RUN apt-get install python3-pip -y
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . ./
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn" , "main:app" , "--host", "0.0.0.0", "--port", "8000"]