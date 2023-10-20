import pickle
from PIL import Image
import matplotlib.pyplot as plt

# Replace 'your_path' with the path where your file is located
file_path = 'selected_images/patch_based_selection_44_selected_images_paths.pkl'


with open(file_path, 'rb') as file:
    data = pickle.load(file)
    for i in range(2):
        image_paths = data[10][i]

        # Plot the images
        for idx, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"Image {idx + 1}")
            plt.axis('off')
            plt.show()

print("end")


