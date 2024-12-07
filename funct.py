import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os
import numpy as np
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array


## all this doese is parse the annotation file i got from the database that adds a red box around the dog 
def parse_ann(file_path):
    
    tree = ET.parse(file_path)
    root = tree.getroot()

    class_name = root.find('.//name').text
    bndbox = root.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    return {'class_name': class_name, 'bbox': (xmin, ymin, xmax, ymax)}

# thiss parits the annotation with the CORRECT image so that box perfectly overlaps the dog in the image
# coll stufff
def pair_files(images, annotation, file_list=None):
   
    pairs = []
    if file_list:
       
        for file in file_list:
            img_path = os.path.join(images, file)
            ann_path = os.path.join(annotation, os.path.splitext(file)[0])

            if os.path.exists(img_path) and os.path.exists(ann_path):
                ann_data = parse_ann(ann_path)
                pairs.append({'image': img_path, 'annotation': ann_data})
    else:
        # Pair directly from the folders
        for root, _, files in os.walk(images):
            for file in files:
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, images)
                ann_path = os.path.join(annotation, os.path.splitext(rel_path)[0])

                if os.path.exists(ann_path):
                    ann_data = parse_ann(ann_path)
                    pairs.append({'image': img_path, 'annotation': ann_data})
    return pairs

# this retruns the number of nested files/images in the image directory)
def count_files(folder):
   
    return sum(len(files) for _, _, files in os.walk(folder))

# shows the pair of dog+ red box. in one image. 
def show_image(img_path, ann):
    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # Draw bounding box
    bbox = ann['bbox']
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)

    # Add class label
    draw.text((bbox[0], bbox[1] - 10), ann['class_name'], fill="red")

    # Display the image
    img.show()

