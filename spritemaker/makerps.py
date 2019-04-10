from pathlib import Path
from PIL import Image
from keras.preprocessing import image
import numpy as np
import joblib

x_train = []
y_train = []
final_image = np.array([])

# "/Users/gantlaborde/Downloads/rps/paper/paper07-119.png"
TRAINING_PATH = "/Users/gantlaborde/Downloads/rps"
SPRITE_SIZE = 64

new_im = Image.new('RGB', (SPRITE_SIZE*SPRITE_SIZE, 2520))

y_offset = 0
# Load the training sprite by looping over every image file
for image_file in Path(TRAINING_PATH).glob("**/*.png"):

    # Load the current image file
    src_image = Image.open(image_file)
    # make it smaller
    downsized = src_image.resize((SPRITE_SIZE,SPRITE_SIZE)) 

    # get 1px high version
    pixels = list(downsized.getdata())
    smoosh = Image.new('RGB', (SPRITE_SIZE * SPRITE_SIZE, 1))
    smoosh.putdata(pixels)

    # build master image 1 pixel down at a time
    new_im.paste(smoosh, (0, y_offset))
    y_offset += 1

    # Use image path to build our answer key
    if "rock" in image_file.stem:
        y_train.append(0) 
    else:
        y_train.append(1)    


new_im.save('data.png')
new_im.show()