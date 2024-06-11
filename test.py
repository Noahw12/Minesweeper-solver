import cv2
from PIL import Image

# def load_image_and_find_average_color(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Check if image is loaded properly
#     if img is None:
#         print("Error: Image could not be read.")
#         return None
    
#     # Calculate the average color
#     # Note: OpenCV loads images in BGR format by default
#     mean_color_per_channel = cv2.mean(img)
    
#     # cv2.mean() returns a tuple of 4 values (B, G, R, Alpha)
#     # If the image has no alpha channel, the fourth value will be 0
#     mean_bgr = mean_color_per_channel[:3]  # We take only the first three values
    
#     print(mean_bgr)

#     return mean_bgr

# # Specify the path to your image

# # Get the average color
# average_color = load_image_and_find_average_color('img/0tile.png')
# average_color1 = load_image_and_find_average_color('img/Blanktile.png')

# if average_color:
#     print("Average color (BGR): 0tile", average_color)
#     print("Average color (BGR): blanktile", average_color1)
    
def split_image(image_path, output_folder):
    # Load the image
    img = Image.open(image_path)
    
    # Dimensions of each tile
    tile_width, tile_height = 16, 16

    # Number of tiles in each dimension
    num_tiles_x = img.width // tile_width
    num_tiles_y = img.height // tile_height

    # Loop over the tiles
    tile_number = 0
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Define the bounding box for the current tile
            left = x * tile_width
            top = y * tile_height
            right = left + tile_width
            bottom = top + tile_height
            box = (left, top, right, bottom)

            # Crop the image to the box
            tile = img.crop(box)

            # Save the tile
            tile.save(f"{output_folder}/tile_{tile_number}.png")
            tile_number += 1

# Path to the sprite sheet
image_path = '2000.png'
output_folder = 'img_pygame'

# Call the function
split_image(image_path, output_folder)