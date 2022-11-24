from google.cloud import storage
import google.cloud.storage
import json
import os
import sys


# PATH = os.path.join("cred.json")
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PATH
#
# storage_client = storage.Client(PATH)
#
# bucket = storage_client.get_bucket('custom-ocr-pytorch')
#
# filename = [filename.name for filename in list(bucket.list_blobs(prefix=''))]
#
# # Downloading a file from Bucket
#
# blop = bucket.blob(blob_name='archive.zip').download_as_string()
#
# with open('archive.zip', 'wb') as f:
#     f.write(blop)


import os


def sync_folder_to_gcloud(filepath, filename, gcp_bucket_url):

    command = f"gsutil cp {filepath}/{filename} gs://{gcp_bucket_url}/"

    command1 = f"gcloud storage cp {filepath}/{filename} gs://{gcp_bucket_url}/"

    os.system(command1)


def sync_folder_from_gcloud(filename, gcp_bucket_url, destination):

    command = f"gsutil cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"

    command1 = f"gcloud storage cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"

    os.system(command1)


sync_folder_to_gcloud("data", "test1.zip", "custom-ocr-pytorch")


sync_folder_from_gcloud("archive.zip", "custom-ocr-pytorch", "data")

