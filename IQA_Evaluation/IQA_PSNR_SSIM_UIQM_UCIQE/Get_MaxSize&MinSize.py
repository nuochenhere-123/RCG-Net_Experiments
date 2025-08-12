import os
from PIL import Image

def get_max_min_image_size(folder_path):
    max_size = (0, 0)
    min_size = None
    max_file = min_file = ""

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            fpath = os.path.join(folder_path, fname)
            with Image.open(fpath) as img:
                width, height = img.size
                if width * height > max_size[0] * max_size[1]:
                    max_size = (width, height)
                    max_file = fname
                if (min_size is None) or (width * height < min_size[0] * min_size[1]):
                    min_size = (width, height)
                    min_file = fname

    print(f"Max size: {max_size} from image: {max_file}")
    print(f"Min size: {min_size} from image: {min_file}")

# 示例用法：
folder = './Test/Test-110'
get_max_min_image_size(folder)
