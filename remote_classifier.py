import json
import struct
import requests
import numpy as np

from functools import singledispatchmethod


class RemoteClassifier:
    def __init__(self, host, port):
        self.__host = host
        self.__port = port
        self.session = requests.Session()

    @staticmethod
    def _array_areas_to_blob(array: np.ndarray, areas: list) -> bytes:
        blob: bytes = b''
        for y, x, l in areas:
            # crop image
            crop: np.ndarray = array[y:y+l, x:x+l, :]
            # grab crop data
            cropdata: bytes = crop.tobytes()
            # encode crop metadata as 16-byte buffer (length, height, width, channels)
            metadata = struct.pack('IIII', len(cropdata), *crop.shape)
            # append metadata and data to blob
            blob += metadata + cropdata
        return blob

    @singledispatchmethod
    def classify(self, img: str):
        with open(img, 'rb') as f:
            r = self.session.post(
                f'http://{self.__host}:{self.__port}/classify', files={'blob': f.read()})
            return json.loads(r.text)

    @classify.register
    def __(self, img: str, areas: list):
        with open(img, 'rb') as f:
            r = self.session.post(
                f'http://{self.__host}:{self.__port}/classify', files={'blob': f.read()})
            return json.loads(r.text)
