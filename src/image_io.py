from PIL import Image


def load_image(path):
    image = Image.open(path)
    return image.convert("RGB")
