from PIL import Image
import os

# Define input and output directories
input_folder = 'img3'
output_folder = 'img'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):  # Add more formats if needed
        # Open the image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Resize the image to 32x32 using high-quality resampling (LANCZOS)
        img_resized = img.resize((32, 32), Image.LANCZOS)

        # Save the resized image to the output folder
        img_resized.save(os.path.join(output_folder, filename), quality=95)  # Adjust quality for JPEGs

print("All images resized and saved with high quality!")
