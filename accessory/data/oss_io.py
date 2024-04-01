import sys
import time
import logging

import cv2
import numpy as np
from PIL import Image
from io import BytesIO


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def load_binary(url):
  global _petrel_client

  return _petrel_client.get(url)


def read_img_general(img_path):
    if "s3://" in img_path:
        init_ceph_client_if_needed()
        img_bytes = client.get(img_path)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        return image
        # cv_img = read_img_ceph(img_path)
        # # return cv_img
        # # noinspection PyUnresolvedReferences
        # # visualization shows next line render the picture blue, remove the cv2.cvtColor
        # return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        # return Image.fromarray(cv_img)
    else:
        return Image.open(img_path).convert('RGB')


client = None


def read_img_ceph(img_path):
    init_ceph_client_if_needed()
    img_bytes = client.get(img_path)
    # # return Image.open(BytesIO(img_bytes)).convert('RGB')
    # assert img_bytes is not None, f"Please check image at {img_path}"
    # img_mem_view = memoryview(img_bytes)
    # img_array = np.frombuffer(img_mem_view, np.uint8)
    # # noinspection PyUnresolvedReferences
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = Image.open(BytesIO(img_bytes)).convert('RGB')
    # return image

    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)

    return image_without_exif


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from data.petrel_client.client import Client  # noqa
        client = Client("/mnt/petrelfs/share_data/linziyi/gaopeng_data/petreloss_all.conf")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")

