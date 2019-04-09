from pathlib import Path
from PIL import Image
from keras.preprocessing import image
import numpy as np
import joblib

x_train = []
y_train = []

# "/Users/gantlaborde/Downloads/rps/paper/paper07-119.png"
TRAINING_PATH = "/Users/gantlaborde/Downloads/rps"
SPRITE_SIZE = 28

# Load the training sprite by looping over every image file
for image_file in Path(TRAINING_PATH).glob("**/*.png"):
    # Load the current image file
    image_data = image.load_img(
        image_file,
        target_size=(SPRITE_SIZE, SPRITE_SIZE)
    )

    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)
    # make 1 pixel high
    smoosh = image_array.reshape(1, SPRITE_SIZE*SPRITE_SIZE, 3)

    # Add the current image 
    x_train.append(smoosh)

    # Use image path to build our answer key
    if "rock" in image_file.stem:
        y_train.append(0) 
    else:
        y_train.append(1)    

img = Image.fromarray(np.concatenate(x_train), 'RGB')
img.save('data.png')
img.show()