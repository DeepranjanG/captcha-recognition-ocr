FROM continuumio/miniconda3

WORKDIR ocr
COPY . /ocr
USER root

RUN apt update -y &&  \
    apt-get update &&  \
    pip install --upgrade pip &&  \
    apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install apt-transport-https ca-certificates gnupg -y
RUN apt-get update && apt-get install google-cloud-cli -y

RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu &&  \
    pip install -r requirements.txt &&  \
    pip install -e .

CMD ["python","app.py"]