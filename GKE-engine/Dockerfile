#Lightweight Python
FROM python:3.7-slim

#Copy local files to the Container Image
WORKDIR /GKE-engine
ADD requirements.txt /GKE-engine/requirements.txt
ADD . /GKE-engine

#Install Dependencies
RUN pip install -r /GKE-engine/requirements.txt
RUN python3 -m laserembeddings download-models

#Run the flask service on container startup
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 wsgi:app
