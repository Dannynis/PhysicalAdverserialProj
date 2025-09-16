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
import tqdm 

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
              
cap = GenericCapturer(url=url)
        

    
border_size = 2

displayed_aruco_code = 40

# Define parameters for ArUco marker detection
aruco_dict_type = cv2.aruco.DICT_6X6_250 # Change dictionary type if needed
marker_length = 0.05  # Marker length in meters (adjust as needed)
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

marker_size = 500  # Size in pixels

proj_marker_image = cv2.aruco.generateImageMarker(aruco_dict, displayed_aruco_code, marker_size)




drawing = False

done = False
ix,iy = -1,-1

rect_corners = None

# define mouse callback function to draw circle
def draw_rectangle(event, x, y, flags, param):
   global ix, iy, drawing, img, done, rect_corners
   if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
      ix = x
      iy = y

   elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
      cv2.rectangle(img, (ix, iy),(x, y),(0, 255, 255),-1)
      rect_corners = [(ix, iy), (x, y)]
      done = True


def capture_many_frames(num_frames=100):
    
    ls = os.listdir('.')
    captures = [f for f in ls if f.startswith('captures_frames_multiview_') ]
    cap_dir = f'captures_frames_multiview_{len(captures)}'
    os.makedirs(cap_dir, exist_ok=True)

    
    for _ in tqdm.tqdm(range(num_frames)):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
        if frame is None:
            continue
        timestamp = int(time.time() * 1000)
        cv2.imwrite(os.path.join(cap_dir, f'frame_{timestamp}.png'), frame)
        
        if not ret:
            break
        
    

screen_res = 1920*2 , 1080*2
# screen_res = 640 , 480
img = np.zeros((screen_res[1], screen_res[0], 3), np.uint8)

to_place = None

orig_proj_striped_corners = None
orig_proj_corners = None
orig_rect_corners = None
width = None
height = None
orig_image = None

def display_drawer():
    global img, done, border_size, to_place, orig_proj_corners, orig_proj_striped_corners,orig_rect_corners
    global width, height, orig_img

    # Create a black image

    # Create a window and bind the function to window
    cv2.namedWindow("Rectangle Window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rectangle Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    # Connect the mouse button to our callback function
    cv2.setMouseCallback("Rectangle Window", draw_rectangle)

    # display the window
    while True:
        cv2.imshow("Rectangle Window", img)
        wk = cv2.waitKey(10)
        if wk == 27 or wk == 99:  # ESC key to exit or 'c' to capture
            break
    # if done:
    #     time.sleep(5)
        
    # cv2.destroyAllWindows()


    img = img[:,:,-1]

    orig_img = img.copy()

    non_zero_indices = np.nonzero(img)

    a,b,c,d = non_zero_indices[0].min(), non_zero_indices[0].max(), non_zero_indices[1].min(), non_zero_indices[1].max()
    width = d - c
    height = b - a



    to_place = cv2.resize(proj_marker_image, (width+1, height+1), interpolation=cv2.INTER_AREA)

    img[img!=0] = to_place.flatten() 

    img[a-border_size:b+border_size, c-border_size:c] = 255

    img[a-border_size:b+border_size, d:d+border_size] = 255

    img[a-border_size:a, c-border_size:d+border_size] = 255

    img[b:b+border_size, c-border_size:d+border_size] = 255


    print('showing')
    # cv2.namedWindow("Rectangle Window aaaa", cv2.WND_PROP_FULLSCREEN)

    # cv2.setWindowProperty("Rectangle Window aaaa", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Rectangle Window", img)

    if wk == 99:
        capture_many_frames(num_frames=100)
    else:
        cv2.waitKey(1)

    orig_rect_corners = [(rect_corners[0][0], rect_corners[0][1]), (rect_corners[1][0], rect_corners[0][1]), (rect_corners[1][0], rect_corners[1][1]), (rect_corners[0][0], rect_corners[1][1])]
    orig_proj_corners = np.array(orig_rect_corners) # np.array([[0, 0], [proj_marker_image.shape[1], 0], [proj_marker_image.shape[1], proj_marker_image.shape[0]], [0, proj_marker_image.shape[0]]], dtype=np.float32)
    orig_proj_striped_corners = np.array([[0, 0], [proj_marker_image.shape[1], 0], [proj_marker_image.shape[1], proj_marker_image.shape[0]], [0, proj_marker_image.shape[0]]], dtype=np.float32)



    


H = None
img_non_zero_section = None

def run_aruco_detector():
    global H, to_place, orig_rect_corners, cap, displayed_aruco_code, img_non_zero_section
    ids = []
    detectorParams = cv2.aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detectorParams)

    detectorParams.adaptiveThreshConstant = 5  # Example: lower the constant
    while ids is None or displayed_aruco_code not in ids :
        for i in range(10):
            ret, frame = cap.read()
            time.sleep(0.1)  # Give some time for the camera to adjust
            if not ret:
                print("Failed to capture image")
                cap = GenericCapturer(url=url)
                continue
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Frame", gray)
            cv2.waitKey(1)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                print("No markers detected, retrying...")
                continue
            else:
                # put the ids on the image
                for i, corner in enumerate(corners):
                    print(ids)


    cv2.destroyAllWindows()

    corners_img_proj = corners[np.where(ids == displayed_aruco_code)[0].item()]

    img_non_zero_section = img[orig_rect_corners[0][1]:orig_rect_corners[2][1], orig_rect_corners[0][0]:orig_rect_corners[1][0]]

    img_non_zero_section_corners = np.array([[0, 0], [img_non_zero_section.shape[1], 0], [img_non_zero_section.shape[1], img_non_zero_section.shape[0]], [0, img_non_zero_section.shape[0]]], dtype=np.float32)

    H, _ = cv2.findHomography(corners_img_proj, img_non_zero_section_corners)

    frame_unwarped = cv2.warpPerspective(frame, H,( img_non_zero_section.shape[1], img_non_zero_section.shape[0]))

    plt.imshow(to_place)
    plt.show()
    plt.imshow(frame_unwarped)
    plt.show()



import tqdm
import IPython
import torch

import torchvision
tpp = torchvision.transforms.ToPILImage()
tp = lambda x: np.array(tpp(x))

def cap_and_uwarp():
    for i in range(1):
        ret, frame = cap.read()
        time.sleep(0.01)
        if not ret:
            print("Failed to capture image")
            break
    frame_unwarped = cv2.warpPerspective(frame, H,( img_non_zero_section.shape[1], img_non_zero_section.shape[0]))
    frame_unwarped = cv2.cvtColor(frame_unwarped, cv2.COLOR_BGR2RGB)
    return frame_unwarped


def plot_on_screen(pimg):

    cv2.namedWindow("Rectangle Window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rectangle Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    to_place = cv2.resize(pimg, (width+1, height+1), interpolation=cv2.INTER_AREA)

    color_pattern = orig_img.copy()

    color_pattern = np.expand_dims(color_pattern, axis=-1).repeat(3, axis=-1) 

    color_pattern[color_pattern!=0] = to_place.flatten()

    color_pattern_BGR = cv2.cvtColor(color_pattern, cv2.COLOR_RGB2BGR)

    cv2.imshow("Rectangle Window", color_pattern_BGR)

    key = cv2.waitKey(1)

    if key == ord('q'):
        raise

    time.sleep(1)

from interp_comp_torch import UltraOptimizedProjectorCompensation5 as UOPC


def photometric_calibration():
    
    proj_wh = 256
    proj_wh = (512, 512)
    low_val = 80 / 255
    high_val = 170 / 255
    n_samples_per_channel = 20
    resizer = torchvision.transforms.Resize((width, height))

    patterns = {
        # "test_image": test_texture,
        "all_black": np.zeros(proj_wh + (3,), dtype=np.float32),
        "off_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "red_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "green_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "blue_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "red_image_2": np.zeros(proj_wh + (3,), dtype=np.float32) * low_val,
        "green_image_2": np.zeros(proj_wh + (3,), dtype=np.float32) * low_val,
        "blue_image_2": np.zeros(proj_wh + (3,), dtype=np.float32) * low_val,
        "red_image_3": np.zeros(proj_wh + (3,), dtype=np.float32) * high_val,
        "green_image_3": np.zeros(proj_wh + (3,), dtype=np.float32) * high_val,
        "blue_image_3": np.zeros(proj_wh + (3,), dtype=np.float32) * high_val,

        "on_image": np.ones(proj_wh + (3,), dtype=np.float32) * high_val,
        "white_image": np.ones(proj_wh + (3,), dtype=np.float32),
    }
    patterns["red_image"][:, :, 0] = high_val
    patterns["green_image"][:, :, 1] = high_val
    patterns["blue_image"][:, :, 2] = high_val
    patterns["red_image_2"][:, :, 0] = high_val
    patterns["green_image_2"][:, :, 1] = high_val
    patterns["blue_image_2"][:, :, 2] = high_val
    patterns["red_image_3"][:, :, 0] = low_val
    patterns["green_image_3"][:, :, 1] = low_val
    patterns["blue_image_3"][:, :, 2] = low_val

    input_values = np.linspace(0.0, 1.0, num=n_samples_per_channel)
    for i in range(n_samples_per_channel):
        patterns["gray_{:03d}".format(i)] = (
            np.ones(proj_wh + (3,), dtype=np.float32) * input_values[i]
        )


    captured = {}
    for description, pattern in tqdm.tqdm(patterns.items()):

            a = torch.from_numpy(pattern).permute(2, 0, 1).float()
            plt.imshow(tp(a))
            plt.show()
            plot_on_screen(tp(a))
            cap = GenericCapturer(url=url)

            time.sleep(0.05)
            unwarped_frames = []
            for i in range(1):
                cur_unwraped = cap_and_uwarp()
                print(cur_unwraped.shape)
                
                unwarped_frames.append(cur_unwraped)
            unwarped_frames = np.array(unwarped_frames)
            frame_unwarped = np.mean(unwarped_frames, axis=0).astype(np.uint8)
            plt.imshow(frame_unwarped)
            plt.show()
            captured[description] = frame_unwarped
            # clear the console
            # time.sleep(0.5)
            IPython.display.clear_output(wait=True)




    anchors_stack = torch.stack([torch.tensor(patterns[key]).float() for key in patterns.keys()]).permute(0,3,1,2)
    name_to_idx = {name: idx for idx, name in enumerate(patterns.keys())}
    projected_anchors = []
    gray_idxs = torch.tensor([name_to_idx[k] for k in name_to_idx if 'gray' in k])
    anchors_gray = resizer(anchors_stack[gray_idxs])
    captured_gray = torch.stack([torch.tensor(captured[k]).float() for k in patterns.keys() if 'gray' in k]).permute(0,3,1,2)