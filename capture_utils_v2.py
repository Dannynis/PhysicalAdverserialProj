import cv2.aruco as aruco
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import imagingcontrol4 as ic4
import numpy as np
import tempfile
import os
import cv2
import tqdm 
from consts import border_size, displayed_aruco_code, marker_size
import torch
import torchvision
from interp_comp_torch import UltraOptimizedProjectorCompensation5 as UOPC

ic4.Library.init()

def bmp_roundtrip(m):
    with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Save to temporary BMP file
        m.save_as_bmp(temp_path)
        
        # Read back with cv2
        cv2_image_bmp = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        cv2_image_bmp = cv2.cvtColor(cv2_image_bmp, cv2.COLOR_BGR2RGB)
        
        return cv2_image_bmp
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class GenericCapturer:
    _ic4_initialized = False
    _global_grab = None
    _global_sink = None

    def __init__(self, url=None):
        if GenericCapturer._ic4_initialized:
            print("IC4 already opened, using existing grabber and sink.")
            self.grabber = GenericCapturer._global_grab
            self.sink = GenericCapturer._global_sink
            self.ic4 = True
            return
        
        # check if ic4 is imported
        if 'ic4' in globals():
            self.ic4 = True
            grabber = ic4.Grabber()

            # Open the first available video capture device
            first_device_info = ic4.DeviceEnum.devices()[0]
            grabber.device_open(first_device_info)
            GenericCapturer._ic4_initialized = True

            print("Pixel format set to RGB8")

            # Create a SnapSink
            sink = ic4.SnapSink()
            grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
            self.grabber = grabber
            self.sink = sink

            GenericCapturer._global_grab = grabber
            GenericCapturer._global_sink = sink
            print("IC4 Grabber and Sink initialized.")
        else:
            cap = cv2.VideoCapture(url)
            self.cap = cap
            self.ic4 = False

    def read(self):
        if self.ic4:
            m = self.sink.snap_single(1000)
            if m is None:
                return None
            cap = bmp_roundtrip(m)
            cap = cv2.resize(cap, (640, 480))
            cap = cv2.cvtColor(cap, cv2.COLOR_RGB2BGR)
            return True, cap
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return ret, frame


class CaptureSystem:
    def __init__(self, url='http://192.168.68.61:8080/video', screen_res=(1920*2, 1080*2)):
        """Initialize the capture system with all parameters as instance variables."""
        # Camera setup
        self.url = url
        self.cap = GenericCapturer(url=self.url)
        
        # Screen and image parameters
        self.screen_res = screen_res
        self.img = np.zeros((screen_res[1], screen_res[0], 3), np.uint8)
        
        # ArUco setup
        self.aruco_dict_type = cv2.aruco.DICT_6X6_250
        self.marker_length = 0.05
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
        self.proj_marker_image = cv2.aruco.generateImageMarker(
            self.aruco_dict, displayed_aruco_code, marker_size
        )
        
        # Drawing state
        self.drawing = False
        self.done = False
        self.ix = -1
        self.iy = -1
        self.rect_corners = None
        
        # Calibration parameters
        self.to_place = None
        self.orig_proj_striped_corners = None
        self.orig_proj_corners = None
        self.orig_rect_corners = None
        self.width = None
        self.height = None
        self.orig_img = None
        self.H = None
        self.img_non_zero_section = None
        
        # Utilities
        self.tpp = torchvision.transforms.ToPILImage()
        self.tp = lambda x: np.array(self.tpp(x))

    def _draw_rectangle(self, event, x, y, flags, param):
        """Mouse callback function for drawing rectangles."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix = x
            self.iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (0, 255, 255), -1)
            self.rect_corners = [(self.ix, self.iy), (x, y)]
            self.done = True

    def display_drawer(self):
        """Interactive display for drawing projection area."""
        cv2.namedWindow("Rectangle Window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Rectangle Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback("Rectangle Window", self._draw_rectangle)

        while True:
            cv2.imshow("Rectangle Window", self.img)
            wk = cv2.waitKey(10)
            if wk == 27 or wk == 99:  # ESC or 'c' key
                break

        self.img = self.img[:, :, -1]
        self.orig_img = self.img.copy()

        non_zero_indices = np.nonzero(self.img)
        a, b, c, d = (non_zero_indices[0].min(), non_zero_indices[0].max(),
                      non_zero_indices[1].min(), non_zero_indices[1].max())
        self.width = d - c
        self.height = b - a

        self.to_place = cv2.resize(self.proj_marker_image, (self.width+1, self.height+1), 
                                   interpolation=cv2.INTER_AREA)
        
        # Resize to account for border size
        self.to_place = cv2.resize(self.to_place, 
                                   (self.width+1 - 2*border_size, self.height+1 - 2*border_size),
                                   interpolation=cv2.INTER_AREA)
        self.to_place = cv2.copyMakeBorder(self.to_place, border_size, border_size, 
                                           border_size, border_size, 
                                           cv2.BORDER_CONSTANT, value=255)

        self.img[self.img != 0] = self.to_place.flatten()

        print('showing')
        cv2.imshow("Rectangle Window", self.img)

        if wk == 99:
            self.capture_many_frames()
        else:
            cv2.waitKey(1)

        self.orig_rect_corners = [
            (self.rect_corners[0][0], self.rect_corners[0][1]),
            (self.rect_corners[1][0], self.rect_corners[0][1]),
            (self.rect_corners[1][0], self.rect_corners[1][1]),
            (self.rect_corners[0][0], self.rect_corners[1][1])
        ]
        self.orig_proj_corners = np.array(self.orig_rect_corners)
        self.orig_proj_striped_corners = np.array([
            [0, 0],
            [self.proj_marker_image.shape[1], 0],
            [self.proj_marker_image.shape[1], self.proj_marker_image.shape[0]],
            [0, self.proj_marker_image.shape[0]]
        ], dtype=np.float32)

    def get_orig_img(self):
        """Return the original image."""
        return self.orig_img

    def capture_many_frames(self):
        """Capture multiple frames and save to disk."""
        ls = os.listdir('./captures_frames_multiview')
        captures = [f for f in ls if f.startswith('captures_frames_multiview_')]
        cap_dir = f'./captures_frames_multiview/captures_frames_multiview_{len(captures)}'
        os.makedirs(cap_dir, exist_ok=True)

        pbar = tqdm.tqdm(total=1000, desc="Capturing frames")
        detectorParams = cv2.aruco.DetectorParameters()
        detector = aruco.ArucoDetector(self.aruco_dict, detectorParams)

        while True:
            if cv2.waitKey(1) == ord('q'):
                break
            ret, frame = self.cap.read()
            frame_copy = frame.copy()

            if frame is None:
                continue
            timestamp = int(time.time() * 1000)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Frame", gray)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                ids_flat = ids.flatten()
                for marker_id, corner in zip(ids_flat, corners):
                    pts = corner.reshape((4, 2)).astype(int)
                    color = (0, 255, 255) if int(marker_id) == displayed_aruco_code else (0, 255, 0)
                    cv2.polylines(frame_copy, [pts], True, color, 2)

            cv2.imshow('frame', frame_copy)
            cv2.imwrite(os.path.join(cap_dir, f'frame_{timestamp}.png'), frame)
            pbar.update(1)

            if not ret:
                break

        pbar.close()

    def run_aruco_detector(self):
        """Detect ArUco markers and compute homography."""
        ids = []
        detectorParams = cv2.aruco.DetectorParameters()
        detector = aruco.ArucoDetector(self.aruco_dict, detectorParams)
        detectorParams.adaptiveThreshConstant = 5

        while ids is None or displayed_aruco_code not in ids:
            for i in range(10):
                ret, frame = self.cap.read()
                time.sleep(0.1)
                if not ret:
                    print("Failed to capture image")
                    self.cap = GenericCapturer(url=self.url)
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
                    for i, corner in enumerate(corners):
                        print(ids)

        cv2.destroyAllWindows()

        # add corners to gray

        corners_img_proj = corners[np.where(ids == displayed_aruco_code)[0].item()]
        self.img_non_zero_section = self.img[
            self.orig_rect_corners[0][1]:self.orig_rect_corners[2][1],
            self.orig_rect_corners[0][0]:self.orig_rect_corners[1][0]
        ]

        # add corners_img_proj to gray
        frame_with_corners = cv2.aruco.drawDetectedMarkers(
            frame.copy(), np.array([corners_img_proj]), np.array([displayed_aruco_code])
        )
        plt.imshow(cv2.cvtColor(frame_with_corners, cv2.COLOR_BGR2RGB))
        plt.show()
        
        img_non_zero_section_corners = np.array([
            [border_size, border_size],
            [self.img_non_zero_section.shape[1] - border_size, border_size],
            [self.img_non_zero_section.shape[1] - border_size, 
             self.img_non_zero_section.shape[0] - border_size],
            [border_size, self.img_non_zero_section.shape[0] - border_size]
        ], dtype=np.float32)

        self.img_non_zero_section_corners = img_non_zero_section_corners

        self.corners_img_proj = corners_img_proj

        self.H, _ = cv2.findHomography(corners_img_proj, img_non_zero_section_corners)

        frame_unwarped = cv2.warpPerspective(
            frame, self.H,
            (self.img_non_zero_section.shape[1], self.img_non_zero_section.shape[0])
        )

        plt.imshow(self.to_place)
        plt.show()
        plt.imshow(frame_unwarped)
        plt.show()

    def cap_and_uwarp(self):
        """Capture and unwarp a frame."""
        for i in range(1):
            ret, frame = self.cap.read()
            time.sleep(0.01)
            if not ret:
                print("Failed to capture image")
                break
        
        frame_unwarped = cv2.warpPerspective(
            frame, self.H,
            (self.img_non_zero_section.shape[1], self.img_non_zero_section.shape[0])
        )
        frame_unwarped = cv2.cvtColor(frame_unwarped, cv2.COLOR_BGR2RGB)
        return frame_unwarped

    def plot_on_screen(self, pimg, back_image=None):
        """Display an image on the projection screen."""
        cv2.namedWindow("Rectangle Window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Rectangle Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if back_image is not None:
            color_pattern = back_image.copy()
        else:
            color_pattern = self.orig_img.copy()

        non_zero_indices = np.nonzero(color_pattern)
        a, b, c, d = (non_zero_indices[0].min(), non_zero_indices[0].max(),
                      non_zero_indices[1].min(), non_zero_indices[1].max())
        width = d - c
        height = b - a
        to_place = cv2.resize(pimg, (width+1, height+1), interpolation=cv2.INTER_AREA)

        color_pattern = np.expand_dims(color_pattern, axis=-1).repeat(3, axis=-1)
        color_pattern[color_pattern != 0] = to_place.flatten()

        color_pattern_BGR = cv2.cvtColor(color_pattern, cv2.COLOR_RGB2BGR)
        cv2.imshow("Rectangle Window", color_pattern_BGR)

        key = cv2.waitKey(1)
        if key == ord('q'):
            raise KeyboardInterrupt

        time.sleep(1)

    def photometric_calibration(self):
        """Perform photometric calibration."""
        proj_wh = (512, 512)
        low_val = 80 / 255
        high_val = 170 / 255
        n_samples_per_channel = 20
        resizer = torchvision.transforms.Resize((self.height, self.width))

        patterns = {
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
            plt.imshow(self.tp(a))
            plt.show()
            self.plot_on_screen(self.tp(a))
            cap = GenericCapturer(url=self.url)

            time.sleep(0.05)
            unwarped_frames = []
            for i in range(1):
                cur_unwraped = self.cap_and_uwarp()
                unwarped_frames.append(cur_unwraped)
            
            unwarped_frames = np.array(unwarped_frames)
            frame_unwarped = np.mean(unwarped_frames, axis=0).astype(np.uint8)
            plt.imshow(frame_unwarped)
            plt.show()
            captured[description] = frame_unwarped

        captured = {k: v.astype(np.float32) / 255.0 for k, v in captured.items()}

        anchors_stack = torch.stack([
            torch.tensor(patterns[key]).float() for key in patterns.keys()
        ]).permute(0, 3, 1, 2)
        
        name_to_idx = {name: idx for idx, name in enumerate(patterns.keys())}
        gray_idxs = torch.tensor([name_to_idx[k] for k in name_to_idx if 'gray' in k])
        anchors_gray = resizer(anchors_stack[gray_idxs])
        captured_gray = torch.stack([
            torch.tensor(captured[k]).float() for k in patterns.keys() if 'gray' in k
        ]).permute(0, 3, 1, 2)

        P = np.stack([
            patterns['red_image'], patterns['green_image'], patterns['blue_image'],
            patterns['red_image_2'], patterns['green_image_2'], patterns['blue_image_2'],
            patterns['red_image_3'], patterns['green_image_3'], patterns['blue_image_3']
        ], axis=0)
        
        C = np.stack([
            captured['red_image'], captured['green_image'], captured['blue_image'],
            captured['red_image_2'], captured['green_image_2'], captured['blue_image_2'],
            captured['red_image_3'], captured['green_image_3'], captured['blue_image_3']
        ], axis=0)

        resized_P = np.stack([cv2.resize(img, (self.width, self.height)) for img in P])

        C_tensor = torch.from_numpy(C)
        P_tensor = torch.from_numpy(resized_P)

        augmentor = UOPC(C_tensor, P_tensor, anchors_gray, captured_gray, device='cpu')

        with open('photometric_calibration.pkl', 'wb') as f:
            pickle.dump({
                'augmentor': augmentor,
                'H': self.H,
                'width': self.width,
                'height': self.height,
                'orig_proj_corners': self.orig_proj_corners,
                'orig_proj_striped_corners': self.orig_proj_striped_corners,
                'orig_rect_corners': self.orig_rect_corners,
            }, f)
