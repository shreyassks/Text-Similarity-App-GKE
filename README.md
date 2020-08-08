# Text Similarity App in Google Cloud Kubernetes Engine
Flask Application deployed in Google Cloud Kubernetes Engine. Consists of a Text Similarity Approach using Multilingual Text Embedding models by Google and Facebook

Facebook's LASER Multilingual Embedding Model and Google's DistilBERT Multilingual Embedding Models have been used in this app to compute the Semantic Similarity between two sentences. The Embeddings generated by these models are used to compute the Cosine Similarity score. 

LASER Embeddings Dimensions      - (1024, 1) 
DistilBERT Embeddings Dimensions - (512 , 1)

**Deployed this app in GCP Kubernetes Engine! But Why? It could be deployed using Heroku!!!!**

I initially tried to deploy the app in Heroku which has the feature to clone the git repo and directly deploy with very little effort. But there is a catch! Heroku allows only if the model size is not greater than **100MB**

Later I tried to deploy using GCP's App Engine with Standard Environment but it was giving strange errors. Remember! The Cloud Storage provided by GCP in default VM is just 5GB!! It can't be customized in Standard Environment. I didn't try with Flexible Environment. It might work!

After finding lot of useful info in wiki **I found out that "LARGE MODELS" should be deployed in Kubernetes Clusters** 

The model size here is very huge because LASER AND BERT models require torch (750 MB). DistilBERT model (600MB) and BERT LARGE NLI (1.24GB) are very large sized models which have to be initialized to generate the Embeddings. 

**Steps to be follwed for Kubernetes Deployment in GCP**

**A) Create the Docker Image and tag it to the google container registry gcr.io**

1. To ignore the cache files while creating Docker Image create a .dockerignore file and put the below text in it
Dockerfile
README.md
*.pyc
*.pyd

2. Set the project
Gcloud config get-value project 
Gcloud config set project <PROJECT-ID>

3. To build a Docker Image and tag it to the Google Container repository (gcr.io)
Gcloud builds submit --tag gcr.io/PROJECT-ID/<giveanametorepository>

4. Once the image is built copy the docker image link and paste it in deployment.yaml file

5. Docker Images will install all the dependancies required for the application

6. Open Container Registry in GCP once the Image is built in console

7. To get additional details of containers built already use gcloud builds 


**B) Deploy the Docker Image in Container Registry to Kubernetes Cluster**

1. To create a new Kubernetes Cluster with Error Reporting enabled
gcloud container clusters create flask-app \
--num-nodes 2 \
--enable-basic-auth \
--issue-client-certificate \
--zone us-west1-b \
--scopes https://www.googleapis.com/auth/logging.write


2. To enable and deploy the application change replica to 1 (enables) from 0 (disables cluster) in deployment.yaml file

3. To run the deployment.yaml file and enable the app
Kubectl apply -f deployment.yaml

4. To run the service.yaml file to generate the External API which can be accessed by public 
Kubectl apply -f service.yaml

5. Some useful commands to know the status of deployment, service and also the pods
Kubectl get deployments 
Kubectl get services
Kubectl get pods

6. To run the External IP in gcloud console and get the output
Curl <External IP Address>
  

**Here is the photo of deployed application in GKE Cluster**

![Text Similarity Application](text%20app.png)
