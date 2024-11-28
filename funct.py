import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import scipy.io

def parse_ann(file_path):
    """
    Parses an annotation file to extract bounding box and class label.

    Parameters:
    - file_path (str): Path to the annotation file.

    Returns:
    - dict: Contains 'class_name' and 'bbox' (bounding box coordinates).
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    class_name = root.find('.//name').text
    bndbox = root.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    return {'class_name': class_name, 'bbox': (xmin, ymin, xmax, ymax)}

def pair_files(images, annotation, file_list):
    """
    Pairs images with their corresponding annotations based on a provided file list.

    Parameters:
    - images (str): Path to the images folder.
    - annotation (str): Path to the annotation folder.
    - file_list (list): List of file paths relative to the images folder.

    Returns:
    - list: A list of paired image and annotation data.
    """
    pairs = []
    for file in file_list:
        img_path = os.path.join(images, file)
        ann_path = os.path.join(annotation, os.path.splitext(file)[0])

        if os.path.exists(img_path) and os.path.exists(ann_path):
            ann_data = parse_ann(ann_path)
            pairs.append({'image': img_path, 'annotation': ann_data})
    return pairs

def count_files(folder):
    """
    Counts the number of files in a folder (recursively).

    Parameters:
    - folder (str): Path to the folder.

    Returns:
    - int: Total number of files.
    """
    return sum(len(files) for _, _, files in os.walk(folder))

def load_mat_list(mat_path):
    """
    Loads a list of file paths from a .mat file.

    Parameters:
    - mat_path (str): Path to the .mat file.

    Returns:
    - list: List of relative file paths.
    """
    mat_data = scipy.io.loadmat(mat_path)
    # Assuming the .mat file contains a variable 'file_list' or similar
    # Adjust 'file_list' based on the actual variable name in the .mat file
    file_list = mat_data.get('file_list', [])
    return [file[0] for file in file_list]

def show_image(img_path, ann):
    """
    Displays an image with its annotation (bounding box and class label).

    Parameters:
    - img_path (str): Path to the image file.
    - ann (dict): Dictionary containing 'class_name' and 'bbox'.
    """
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # Draw bounding box
    bbox = ann['bbox']
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)

    # Add class label
    draw.text((bbox[0], bbox[1] - 10), ann['class_name'], fill="red")

    # Display the image
    img.show()
