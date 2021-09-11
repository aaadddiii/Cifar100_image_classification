import io, base64
from PIL import Image

def decode_image(base_64_image):
    """Decode image from base64 to numpy array"""
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_image, "utf-8"))))
    return img