import cv2.aruco as aruco
import pickle
import time

import cv2
import matplotlib.pyplot as plt

url = 'http://192.168.68.61:8080/video'
# url = 'http://localhost:8080/video_feed'

import imagingcontrol4 as ic4
import numpy as np
import tempfile
import os
import cv2

ic4.Library.init()



def bmp_roundtrip(m):

    with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Save to temporary BMP file
        m.save_as_bmp(temp_path)
        
        # Read back with cv2
        cv2_image_bmp = cv2.imread(temp_path, cv2.IMREAD_COLOR)

        cv2_image_bmp = cv2.cvtColor(cv2_image_bmp, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed
        
        return cv2_image_bmp
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)



