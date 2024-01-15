import imageio.v2 as imageio
import os
from PIL import Image
import numpy as np

# Function to resize images to be divisible by the macro block size
def resize_image(image_path, macro_block_size=16):
    img = Image.open(image_path)
    w, h = img.size
    w_new = w - w % macro_block_size
    h_new = h - h % macro_block_size
    # You could also add padding instead of resizing, depending on your preference.
    img = img.resize((w_new, h_new), Image.ANTIALIAS)
    return np.array(img)

# The folder containing your PNG files
folder_path = './Check_plots/'  # Change to your folder path

# Sort the file names (assuming file names are sortable and represent sequence order)
file_names = sorted((fn for fn in os.listdir(folder_path) if fn.endswith('.png')))

# Define the path for the output video file
output_path = 'zanimation.mp4'

# Create a writer object specifying the fps (frames per second)
writer = imageio.get_writer(output_path, fps=10)

# Loop through the sorted file names and add each resized image to the video
for filename in file_names:
    image_path = os.path.join(folder_path, filename)
    image = resize_image(image_path)
    writer.append_data(image)

# Close the writer to finish writing the video file
writer.close()

print("Video created successfully at:", output_path)

