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
    save_dir = "data"  # Change this to your desired directory

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Full path for the output file
    output_file = os.path.join(save_dir, "imagenet_sketch.zip")  # Adjust file name as needed

    # Download the file
    gdown.download(direct_link, output_file, quiet=False)

    print(f"Downloaded file saved as: {output_file}")

def extract_imagenet_sketch():
    save_dir = "data"
    # Full path for the output file
    output_file = os.path.join(save_dir, "imagenet_sketch.zip")  # Adjust file name as needed

    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def extract_imagenet_a():
    # Directory where the downloaded file is saved
    save_dir = "data"

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
    save_dir = "data"  # Change this to your desired directory

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
    save_dir = "data"

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
    save_dir = "data"  # Change this to your desired directory

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



def extract_imagenet_c_extra():
    # Directory where the downloaded file is saved
    

    save_dir = "data/ImageNet-C"

    # Full path for the `.tar` file
    output_file = os.path.join(save_dir, "extra.tar")  # Adjust file name as needed

    # Extract the tar file
    with tarfile.open(output_file, 'r') as tar_ref:
        tar_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def download_imagenet_c_extra():
    # Direct download link
    direct_download_link = "https://zenodo.org/records/2235448/files/extra.tar?download=1"  # Replace with your actual URL

    # Define the folder path
    base_folder = "data"
    new_folder_name = "ImageNet-C"
    new_folder_path = os.path.join(base_folder, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)

    save_dir = new_folder_path

    # Full path for the output file
    output_file = os.path.join(save_dir, "extra.tar")  # Adjust file name as needed

    # Download the file
    response = requests.get(direct_download_link, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved as: {output_file}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")

def extract_imagenet_c_weather():
    # Directory where the downloaded file is saved
    

    save_dir = "data/ImageNet-C"

    # Full path for the `.tar` file
    output_file = os.path.join(save_dir, "weather.tar")  # Adjust file name as needed

    # Extract the tar file
    with tarfile.open(output_file, 'r') as tar_ref:
        tar_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def download_imagenet_c_weather():
    # Direct download link
    direct_download_link = "https://zenodo.org/records/2235448/files/weather.tar?download=1"  # Replace with your actual URL

    # Define the folder path
    base_folder = "data"
    new_folder_name = "ImageNet-C"
    new_folder_path = os.path.join(base_folder, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)

    save_dir = new_folder_path

    # Full path for the output file
    output_file = os.path.join(save_dir, "weather.tar")  # Adjust file name as needed

    # Download the file
    response = requests.get(direct_download_link, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved as: {output_file}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")



def extract_imagenet_c_digital():
    # Directory where the downloaded file is saved
    

    save_dir = "data/ImageNet-C"

    # Full path for the `.tar` file
    output_file = os.path.join(save_dir, "digital.tar")  # Adjust file name as needed

    # Extract the tar file
    with tarfile.open(output_file, 'r') as tar_ref:
        tar_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def download_imagenet_c_digital():
    # Direct download link
    direct_download_link = "https://zenodo.org/records/2235448/files/digital.tar?download=1"  # Replace with your actual URL

    # Define the folder path
    base_folder = "data"
    new_folder_name = "ImageNet-C"
    new_folder_path = os.path.join(base_folder, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)

    save_dir = new_folder_path

    # Full path for the output file
    output_file = os.path.join(save_dir, "digital.tar")  # Adjust file name as needed

    # Download the file
    response = requests.get(direct_download_link, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved as: {output_file}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")



def extract_imagenet_c_blur():
    # Directory where the downloaded file is saved
    

    save_dir = "data/ImageNet-C"

    # Full path for the `.tar` file
    output_file = os.path.join(save_dir, "blur.tar")  # Adjust file name as needed

    # Extract the tar file
    with tarfile.open(output_file, 'r') as tar_ref:
        tar_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def download_imagenet_c_blur():
    # Direct download link
    direct_download_link = "https://zenodo.org/records/2235448/files/blur.tar?download=1"  # Replace with your actual URL

    # Define the folder path
    base_folder = "data"
    new_folder_name = "ImageNet-C"
    new_folder_path = os.path.join(base_folder, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)

    save_dir = new_folder_path

    # Full path for the output file
    output_file = os.path.join(save_dir, "blur.tar")  # Adjust file name as needed

    # Download the file
    response = requests.get(direct_download_link, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved as: {output_file}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")



def extract_imagenet_c_noise():
    # Directory where the downloaded file is saved
    

    save_dir = "data/ImageNet-C"

    # Full path for the `.tar` file
    output_file = os.path.join(save_dir, "noise.tar")  # Adjust file name as needed

    # Extract the tar file
    with tarfile.open(output_file, 'r') as tar_ref:
        tar_ref.extractall(save_dir)  # Extract to the same directory
    print(f"Extracted files to: {save_dir}")


def download_imagenet_c_noise():
    # Direct download link
    direct_download_link = "https://zenodo.org/records/2235448/files/noise.tar?download=1"  # Replace with your actual URL

    # Define the folder path
    base_folder = "data"
    new_folder_name = "ImageNet-C"
    new_folder_path = os.path.join(base_folder, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)

    save_dir = new_folder_path

    # Full path for the output file
    output_file = os.path.join(save_dir, "noise.tar")  # Adjust file name as needed

    # Download the file
    response = requests.get(direct_download_link, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file saved as: {output_file}")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")

<<<<<<< HEAD
download_imagenet_a_flag = False
=======
download_imagenet_a_flag = True
>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332
if download_imagenet_a_flag:
    download_imagenet_a()
    extract_imagenet_a()

download_imagenet_r_flag = False
if download_imagenet_r_flag:
    download_imagenet_r()
    extract_imagenet_r()

download_imagenet_sketch_flag = False
if download_imagenet_sketch_flag:
    download_imagenet_sketch()
    extract_imagenet_sketch()


<<<<<<< HEAD
download_imagenet_c_noise_flag = True
=======
download_imagenet_c_noise_flag = False
>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332
if download_imagenet_c_noise_flag:
    download_imagenet_c_noise()
    extract_imagenet_c_noise()



<<<<<<< HEAD
download_imagenet_c_blur_flag = True
=======
download_imagenet_c_blur_flag = False
>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332
if download_imagenet_c_blur_flag:
    download_imagenet_c_blur()
    extract_imagenet_c_blur()



download_imagenet_c_extra_flag = False
if download_imagenet_c_extra_flag:
    download_imagenet_c_extra()
    extract_imagenet_c_extra()



download_imagenet_c_weather_flag = False
if download_imagenet_c_weather_flag:
    download_imagenet_c_weather()
    extract_imagenet_c_weather()



download_imagenet_c_digital_flag = False
if download_imagenet_c_digital_flag:
    download_imagenet_c_digital()
    extract_imagenet_c_digital()
