import os
import gdown

import os
import requests


import tarfile
import os

import zipfile
import os

def download_imagenet_sketch():
    # Google Drive link
    google_drive_link = "https://drive.usercontent.google.com/download?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA&export=download&authuser=0"

    # Extract the file ID from the link
    file_id = google_drive_link.split('/d/')[1].split('/')[0]

    # Generate a direct download link
    direct_link = f"https://drive.google.com/uc?id={file_id}"

    # Directory where you want to save the file
    save_dir = '../test-time-adaptation/data/'  # Change this to your desired directory

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Full path for the output file
    output_file = os.path.join(save_dir, "imagenet_sketch.zip")  # Adjust file name as needed

    # Download the file
    gdown.download(direct_link, output_file, quiet=False)

    print(f"Downloaded file saved as: {output_file}")

def extract_imagenet_sketch():
    save_dir = '../test-time-adaptation/data/'
    # Full path for the output file
    output_file = os.path.join(save_dir, "imagenet_sketch.zip")  # Adjust file name as needed

    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def extract_imagenet_a():
    # Directory where the downloaded file is saved
    save_dir = '../test-time-adaptation/data/'

    # Full path for the `.tar` file
    output_file = os.path.join(save_dir, "imagenet-a.tar")  # Adjust file name as needed

    # Extract the tar file
    with tarfile.open(output_file, 'r') as tar_ref:
        tar_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def download_imagenet_a():
    # Direct download link
    direct_download_link = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"  # Replace with your actual URL

    # Directory where you want to save the file
    save_dir = '../test-time-adaptation/data/'  # Change this to your desired directory

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Full path for the output file
    output_file = os.path.join(save_dir, "imagenet-a.tar")  # Adjust file name as needed

    # Download the file
    response = requests.get(direct_download_link, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved as: {output_file}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")


def extract_imagenet_r():
    # Directory where the downloaded file is saved
    save_dir = '../test-time-adaptation/data/'

    # Full path for the `.tar` file
    output_file = os.path.join(save_dir, "imagenet-r.tar")  # Adjust file name as needed

    # Extract the tar file
    with tarfile.open(output_file, 'r') as tar_ref:
        tar_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def download_imagenet_r():
    # Direct download link
    direct_download_link = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"  # Replace with your actual URL

    # Directory where you want to save the file
    save_dir = '../test-time-adaptation/data/'  # Change this to your desired directory

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Full path for the output file
    output_file = os.path.join(save_dir, "imagenet-r.tar")  # Adjust file name as needed

    # Download the file
    response = requests.get(direct_download_link, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved as: {output_file}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")



download_imagenet_a_flag = True
if download_imagenet_a_flag:
    download_imagenet_a()
    extract_imagenet_a()

download_imagenet_r_flag = True
if download_imagenet_r_flag:
    download_imagenet_r()
    extract_imagenet_r()

download_imagenet_sketch_flag = True
if download_imagenet_sketch_flag:
    download_imagenet_sketch()
    extract_imagenet_sketch()

