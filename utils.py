from PIL import Image, ImageOps
import numpy as np
import os

FONTS_PATH = 'fonts/'

def preprocess_font_names(fonts_path):
    """
    Shortens file names
    :param fonts_path: path, where fonts are stored
    :return:
    """
    for font in os.listdir(fonts_path):
        for digit in os.listdir(f"{fonts_path}{font}/"):
            source = f"{fonts_path}{font}/{digit}"
            dest = f"{fonts_path}{font}/{digit[-6:]}"
            os.rename(source, dest)


def preprocess_font_images(fonts_path):
    """
    Crops digits images to remove margins
    :param fonts_path: path, where fonts are stored
    :return:
    """
    for font in os.listdir(fonts_path):
        for digit in os.listdir(f"{fonts_path}{font}/"):
            img = np.asarray(ImageOps.invert(ImageOps.grayscale(Image.open(f"{fonts_path}{font}/{digit}"))))/255.0

            nonzero = np.where(img > 0)
            top = min(nonzero[0])
            bottom = max(nonzero[0])+1
            left = min(nonzero[1])
            right = max(nonzero[1])+1
            print(top, bottom, left, right)
            img = Image.fromarray(np.uint8(img*255))
            img = img.crop((left, top, right, bottom))
            img.save(f"{fonts_path}{font}/{digit}")


if __name__ == '__main__':
    # preprocess_font_names(fonts_path=cfg.FONTS_PATH)
    preprocess_font_images(fonts_path=FONTS_PATH)

