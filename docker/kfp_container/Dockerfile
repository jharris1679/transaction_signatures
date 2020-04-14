FROM ubuntu:18.04
#FROM google/cloud-sdk:latest

WORKDIR /home/

RUN apt-get update -y && \
     apt-get -y upgrade && \
     apt-get install -y python3-pip python3-dev && \
     apt-get install -y libgtk2.0-dev && \
     apt-get install -y software-properties-common && \
     apt-get install -y vim

# download and install gcloud
RUN  export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
     echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
     apt-get update && apt-get install -y google-cloud-sdk

ENV LC_CTYPE=C.UTF-8

COPY payload/requirements.txt /srv/payload/
RUN pip3 install --requirement /srv/payload/requirements.txt

COPY app/ /home/

# Configure google cloud tools
RUN gcloud config set account powerups-personalization@tensile-oarlock-191715.iam.gserviceaccount.com --installation && \
    gcloud config set project tensile-oarlock-191715 --installation && \
    gcloud auth activate-service-account --key-file=/home/keydir/pp_key.json && \
    gcloud container clusters get-credentials kubeflow-pipeline --zone us-central1-c #--project tensile-oarlock-191715


ENV GOOGLE_APPLICATION_CREDENTIALS='/home/keydir/pp_key.json'

ARG EXP_ID
RUN echo $EXP_ID
ENV EXP_ID=$EXP_ID

ENTRYPOINT ["python3", "embed_users_pipeline.py"]

#CMD ["bash"]
