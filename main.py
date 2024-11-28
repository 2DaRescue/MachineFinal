import random
from funct import pair_files, show_image, count_files, load_mat_list


######################
# getting the data.  
#######################

# Paths to images and annotations
images = './Data_set/Images'
annotation = './Data_set/Annotation'

train_mat_path = './Data_set/List/train_list.mat'
test_mat_path = './Data_set/List/test_list.mat'

#count number of files 
print(f"Files in images folder {count_files(images)}" )
print(f"files in annotation folder {count_files(annotation)}" )


# Pair the files
data = pair_files(images, annotation)

# Display a random image + annotation
random_pair = random.choice(data)  # Select a random pair
show_image(random_pair['image'], random_pair['annotation'])


#####################                                          #####################  
#####################    splitting the data . test and train.  #####################  
#####################                                          #####################  

# Step 2: Load and Split Dataset

# Load train and test lists from .mat files
train_list = load_mat_list(train_mat_path)
test_list = load_mat_list(test_mat_path)

# Pair files for train and test sets
train_data = pair_files(images, annotation, train_list)
test_data = pair_files(images, annotation, test_list)

# Output summary
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
