# -*- coding: utf-8 -*-
"""helmet_det.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PdwU1OAHKazhgNZqg_ErF5YCnpYHXFkn
"""

from google.colab import drive
drive.mount('/content/gdrive')

ROOT_DIR ='/content/gdrive/My Drive/new_augmented_data_helmet'

!pip install ultralytics
!pip install PyDrive

import yaml

# Define the data to be written in the YAML file
data = {
    'path': '/content/gdrive/My Drive/new_augmented_data_helmet',
    'train': 'images/train',
    'val': 'images/test',
    'names': {
        0: 'motorbike',
        1: 'DHelmet',
        2: 'DNoHelmet',
        3: 'P1Helmet',
        4: 'P1NoHelmet',
        5: 'P2Helmet',
        6: 'P2NoHelmet'
    }
}

# Specify the file path and name
file_path = '/content/gdrive/My Drive/new_augmented_data_helmet/google_colab_file_helmet.yaml'
with open(file_path, 'w') as file:
    yaml.dump(data, file)

!grep -rni "^7" '/content/gdrive/My Drive/helmet_data_iit_mandi_project/labels/train' | wc -l

"""#For Storing particular class images in a different folder -

"""

import os
import shutil

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def copy_files_starting_with_zero(source_image_folder, source_txt_folder, destination_image_folder, destination_txt_folder):
    for txt_filename in os.listdir(source_txt_folder):
        if txt_filename.endswith(".txt"):
            source_txt_file_path = os.path.join(source_txt_folder, txt_filename)
            with open(source_txt_file_path, 'r') as txt_file:
                lines = txt_file.readlines()
                for line in lines:
                    if line.strip().startswith("3"):
                        # Find the corresponding image filename
                        image_filename = txt_filename[:-4] + '.jpg'  # Assuming the image has a .jpg extension

                        # Copy the .txt file to the destination folder
                        destination_txt_file_path = os.path.join(destination_txt_folder, txt_filename)
                        create_folder_if_not_exists(destination_txt_folder)
                        shutil.copyfile(source_txt_file_path, destination_txt_file_path)

                        # Copy the image to the destination folder
                        source_image_path = os.path.join(source_image_folder, image_filename)
                        destination_image_path = os.path.join(destination_image_folder, image_filename)
                        if os.path.exists(source_image_path):
                            create_folder_if_not_exists(destination_image_folder)
                            shutil.copyfile(source_image_path, destination_image_path)
                        break  # Break the loop if we found a line starting with "6" to avoid copying multiple times

source_image_folder = '/content/gdrive/My Drive/helmet_data_iit_mandi_project/images/train'
source_txt_folder = '/content/gdrive/My Drive/helmet_data_iit_mandi_project/labels/train'
destination_image_folder = '/content/gdrive/My Drive/helmet_data_iit_mandi_project/augimages/augimg3'
destination_txt_folder = '/content/gdrive/My Drive/helmet_data_iit_mandi_project/augimages/auglabel3'

copy_files_starting_with_zero(source_image_folder, source_txt_folder, destination_image_folder, destination_txt_folder)

"""Data Augmentation"""

!pip install Augmentor

import os
import Augmentor

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def augment_images(source_image_folder, destination_image_folder, num_augmented_images):
    # Create an Augmentor pipeline for image augmentation
    pipeline = Augmentor.Pipeline(source_image_folder, output_directory=destination_image_folder)

    # Add augmentation operations to the pipeline (you can customize these as per your requirements)
    pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.5)
    pipeline.zoom_random(probability=0.5, percentage_area=0.8)
    pipeline.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)

    # Generate augmented images
    pipeline.sample(num_augmented_images)

source_image_folder = '/content/gdrive/My Drive/helmet_data_iit_mandi_project/augimages/augimg4'
destination_image_folder = '/content/gdrive/My Drive/helmet_data_iit_mandi_project/result_aug/result_augimg4'
num_augmented_images = 200

augment_images(source_image_folder, destination_image_folder, num_augmented_images)

"""#For deletion of extra or dublicate anotations -"""

import os

def check_and_delete_extra_annotations(image_folder, annotation_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    annotation_files = [f for f in os.listdir(annotation_folder) if f.lower().endswith('.txt')]

    # Find annotations without corresponding images and delete them
    for annotation_file in annotation_files:
        image_name = annotation_file.replace('.txt', '.jpg')
        if image_name not in image_files:
            annotation_path = os.path.join(annotation_folder, annotation_file)
            os.remove(annotation_path)
            print(f"Deleted {annotation_file} as it has no corresponding image.")

# Replace these paths with the actual paths to your image and annotation folders
image_folder = '/content/gdrive/My Drive/new_augmented_data_helmet/images/train'
annotation_folder = '/content/gdrive/My Drive/new_augmented_data_helmet/labels/train'

check_and_delete_extra_annotations(image_folder, annotation_folder)

"""#Model Training -"""

import os
from ultralytics import YOLO

#load model
model= YOLO("yolov8n.yaml")
#!yolo predict model = yolov8n.pt source="/content/gdrive/My Drive/carnoplate/images/train/download (1).jpeg"

results = model.train(data =os.path.join(ROOT_DIR ,"google_colab_file_helmet.yaml") ,epochs =100)

"""Saving the results -"""

!scp -r /content/runs '/content/gdrive/My Drive/new_augmented_data_helmet'
