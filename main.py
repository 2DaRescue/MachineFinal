import random
from funct import pair_files, show_image

# Paths to images and annotations
images = './Data_set/Images'
anns = './Data_set/Annotation'

# Pair the files
data = pair_files(images, anns)

# Display a random image with its annotation
if data:
    random_pair = random.choice(data)  # Select a random pair
    print(f"Displaying a random image with annotation:")
    print(f"Image: {random_pair['image']}, Annotation: {random_pair['annotation']}")
    show_image(random_pair['image'], random_pair['annotation'])
else:
    print("No data to display.")
