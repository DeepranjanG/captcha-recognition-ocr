FROM ubuntu:20.04

RUN apt-get update -y \
&& apt-get install -y software-properties-common \
&& add-apt-repository ppa:deadsnakes/ppa \
&& apt-get install openjdk-8-jdk -y \
&& apt-get install python3-pip -y \
&& apt-get install ffmpeg libsm6 libxext6  -y


RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-410.0.0-linux-x86_64.tar.gz

ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/
RUN pip3 install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install -e .

RUN airflow db init
RUN airflow users create  -e deepranjan@ineuron.ai -f Deepranjan -l Gupta -p admin -r Admin  -u admin
RUN chmod 777 start.sh
ENTRYPOINT [ "/bin/sh" ]
CMD ["start.sh"]