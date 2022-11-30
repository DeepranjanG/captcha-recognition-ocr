FROM continuumio/miniconda3

WORKDIR ocr
COPY . /ocr

RUN apt update -y &&  \
    apt-get update &&  \
    pip install --upgrade pip &&  \
    apt-get install ffmpeg libsm6 libxext6  -y \

RUN apt-get install apt-transport-https ca-certificates gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install google-cloud-cli

RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu &&  \
    pip install -r requirements.txt &&  \
    pip install -e .

CMD ["python","app.py"]