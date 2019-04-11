from pathlib import Path
from PIL import Image
import numpy as np

# Constants
TRAINING_PATH = "/Users/gantlaborde/Downloads/rps"
SPRITE_SIZE = 64

# Initialization
x_data = []
y_data = []
final_image = np.array([])
y_offset = 0
new_im = Image.new('RGB', (SPRITE_SIZE*SPRITE_SIZE, 2520))

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

    # store image
    x_data.append(smoosh)

    # Use image path to build our answer key
    if "rock" in image_file.stem:
        y_data.append(1)
    elif "paper" in image_file.stem:
        y_data.append(2)
    else:
        y_data.append(3)
   
# Now randomize X and Y the same way before making data
# (the JS code splits then randomizes) DERP!!!
assert len(y_data) == len(x_data)
p = np.random.permutation(len(y_data))
npy = np.array(y_data)
shuffled_y = npy[p].tolist()

one_hot_y = []
# Build the data image and 1-hot encoded answer array
for idx in p:
    # build master sprite 1 pixel down at a time
    new_im.paste(x_data[idx], (0, y_offset))
    y_offset += 1
    # build 1-hot encoded answer key
    if shuffled_y[idx] == 1:
        one_hot_y.append(1)
        one_hot_y.append(0)
        one_hot_y.append(0)
    elif shuffled_y[idx] == 2:
        one_hot_y.append(0)
        one_hot_y.append(1)
        one_hot_y.append(0)
    else:
        one_hot_y.append(0)
        one_hot_y.append(0)
        one_hot_y.append(1)


# Save answers file (Y)
newFile = open("labels_uint8", "wb")
newFileByteArray = bytearray(one_hot_y)
bytesWritte = newFile.write(newFileByteArray)
# should be num classes * original answer key size
assert bytesWritte == (3 * len(y_data))



# Save Data Sprite (X)
new_im.save('data.png')
# new_im.show() # For debugging