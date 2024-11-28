import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

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

def pair_files(images, anns):
    """
    Pairs images with their corresponding annotations.

    Parameters:
    - images (str): Path to the images folder.
    - anns (str): Path to the annotations folder.

    Returns:
    - list: A list of paired image and annotation data.
    """
    pairs = []
    for root, _, files in os.walk(images):
        for file in files:
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, images)
            ann_path = os.path.join(anns, os.path.splitext(rel_path)[0])

            if os.path.exists(ann_path):
                ann_data = parse_ann(ann_path)
                pairs.append({'image': img_path, 'annotation': ann_data})
    return pairs

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
