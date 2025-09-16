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



opened_ic4 = False
global_grab = None
global_sink = None

class GenericCapturer:
    def __init__(self, url=None):
        global opened_ic4, global_grab, global_sink

        if opened_ic4:
            print("IC4 already opened, using existing grabber and sink.")
            self.grabber = global_grab
            self.sink = global_sink
            self.ic4 = True
            return
        # check if ic4 is imported
        if 'ic4' in globals():
            # Create a Grabber object
            self.ic4 = True
            grabber = ic4.Grabber()


            # Open the first available video capture device
            first_device_info = ic4.DeviceEnum.devices()[0]
            grabber.device_open(first_device_info)
            opened_ic4 = True

            # grabber.device_property_map.set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.BGR8     )
            # Set the resolution to 640x480
            # grabber.device_property_map.set_value(ic4.PropId.WIDTH, 640)
            # grabber.device_property_map.set_value(ic4.PropId.HEIGHT, 480)

            # Set pixel format to RGB8 for color images
            # grabber.device_property_map.set_value(ic4.PropId.PIXEL_FORMAT,    ic4.PixelFormat.BayerRG8)
            print("Pixel format set to RGB8")


            # Create a SnapSink. A SnapSink allows grabbing single images (or image sequences) out of a data stream.
            sink = ic4.SnapSink()
            # Setup data stream from the video capture device to the sink and start image acquisition.
            grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
            self.grabber = grabber
            self.sink = sink

            global_grab = grabber
            global_sink = sink
            print("IC4 Grabber and Sink initialized.")

        else:
            cap = cv2.VideoCapture(url)

            camera_index = 0
            # cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) # this is the magic!
            self.cap = cap
            self.ic4 = False

    def read(self):
        if self.ic4:
            # Read a single image from the Grabber
            m = self.sink.snap_single(1000)
            if m is None:
                return None
            cap = bmp_roundtrip(m)
            # resize the image to 640x480
            cap = cv2.resize(cap, (640, 480))
            cap = cv2.cvtColor(cap, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV compatibility
            return True, cap
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return ret, frame        
        

    
border_size = 2

def gen_display_aruco_marker():
    displayed_aruco_code = 40

    # Define parameters for ArUco marker detection
    aruco_dict_type = cv2.aruco.DICT_6X6_250 # Change dictionary type if needed
    marker_length = 0.05  # Marker length in meters (adjust as needed)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

    marker_size = 500  # Size in pixels

    proj_marker_image = cv2.aruco.generateImageMarker(aruco_dict, displayed_aruco_code, marker_size)


    return proj_marker_image


