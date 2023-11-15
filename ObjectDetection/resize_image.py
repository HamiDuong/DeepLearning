from PIL import Image
import os

"Resizes all images in the given directory - keep ratio and image is fitted into the given numbers"
def resize(dic_path, width, height):
    for file in os.listdir(dic_path):
        img = Image.open(os.path.join(dic_path, file))
        img.thumbnail((width,height))
        img.save(os.path.join(dic_path, file))

path = "Fullscreen_Screenshots_1024x576"
width = 1024
height = 1024

resize(path, width, height)