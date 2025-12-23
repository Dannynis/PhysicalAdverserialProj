import IPython
import cv2.aruco as aruco
import pickle
import time

import datetime
import cv2
import matplotlib.pyplot as plt
import imagingcontrol4 as ic4
import numpy as np
import tempfile
import os
import cv2
import tqdm 
from consts import border_size, displayed_aruco_code, marker_size, aruco_dict_type
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
        self.aruco_dict_type = aruco_dict_type
        self.marker_length = 0.05
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
        self.proj_marker_image = cv2.aruco.generateImageMarker(
            self.aruco_dict, displayed_aruco_code, marker_size
        )
        # trim proj_marker_image brightness to 200
        self.proj_marker_image = (self.proj_marker_image.astype(np.float32) / 255.0 * 200).astype(np.uint8)
        
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

        # self.orig_rect_corners = [
        #     (self.rect_corners[0][0], self.rect_corners[0][1]),
        #     (self.rect_corners[1][0], self.rect_corners[0][1]),
        #     (self.rect_corners[1][0], self.rect_corners[1][1]),
        #     (self.rect_corners[0][0], self.rect_corners[1][1])
        # ]
        
        xs = np.array(self.rect_corners)[:,0]
        ys = np.array(self.rect_corners)[:,1]

        self.orig_rect_corners = [[xs.min(), ys.min()],
                   [xs.max(), ys.min()],
                     [xs.max(), ys.max()],
                        [xs.min(), ys.max()]]
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
        # Enable corner refinement for tighter corner detection
        detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detectorParams.cornerRefinementWinSize = 5
        detectorParams.cornerRefinementMaxIterations = 50
        detectorParams.cornerRefinementMinAccuracy = 0.01
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
                    cv2.imwrite(os.path.join(cap_dir, f'frame_{timestamp}.png'), frame)

            cv2.imshow('frame', frame_copy)
            pbar.update(1)

            if not ret:
                break

        pbar.close()

    def random_contrast_adjustment(self, image, alpha=None):
        """Apply random contrast adjustment to the image."""
        if alpha is None:
            alpha = np.random.uniform(0.5, 1)  # Contrast control

        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

        return adjusted, alpha
    def run_aruco_detector(self):
        """Detect ArUco markers and compute homography."""
        ids = []
        best_alpha = None
        detectorParams = cv2.aruco.DetectorParameters()
        
        # Enable corner refinement for more accurate corner detection
        detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detectorParams.cornerRefinementWinSize = 5  # Window size for corner refinement
        detectorParams.cornerRefinementMaxIterations = 50  # Max iterations for refinement
        detectorParams.cornerRefinementMinAccuracy = 0.01  # Stop when accuracy is below this
        
        # Adaptive threshold parameters for better edge detection
        detectorParams.adaptiveThreshConstant = 7
        detectorParams.adaptiveThreshWinSizeMin = 3
        detectorParams.adaptiveThreshWinSizeMax = 23
        detectorParams.adaptiveThreshWinSizeStep = 10
        
        # Minimum marker perimeter rate (relative to image size)
        detectorParams.minMarkerPerimeterRate = 0.01
        detectorParams.maxMarkerPerimeterRate = 4.0
        
        # Polygonal approximation accuracy
        detectorParams.polygonalApproxAccuracyRate = 0.03
        
        detector = aruco.ArucoDetector(self.aruco_dict, detectorParams)
        detected_corners = []
        marked_frames = 5
        while ids is None or len(detected_corners) < marked_frames:
            IPython.display.clear_output(wait=True)
            print(f"{len(detected_corners)}/{marked_frames} marker corners detected.")
            for i in range(3):
                ret, frame = self.cap.read()
                time.sleep(0.01)
                if not ret:
                    print("Failed to capture image")
                    self.cap = GenericCapturer(url=self.url)
                    continue
            
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray, alpha = self.random_contrast_adjustment(gray, best_alpha)
                
                cv2.imshow("Frame", gray)
                cv2.waitKey(1)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is None or displayed_aruco_code not in ids.flatten():
                    print("No markers detected, retrying...")
                    best_alpha = None
                    continue
                else:
                    print(ids)
                    best_alpha = alpha
                    for i, corner in enumerate(corners):
                        if ids[i] == displayed_aruco_code:
                            detected_corners.append(corner)
                
           

        dc = np.array(detected_corners)
        center = np.floor(dc.reshape(-1, 2).mean(axis=0))
        tl = ((dc - center)**2).sum(axis=-1)[:,0,0].argmin()
        tr = ((dc - center)**2).sum(axis=-1)[:,0,1].argmin()
        br = ((dc - center)**2).sum(axis=-1)[:,0,2].argmin()
        bl = ((dc - center)**2).sum(axis=-1)[:,0,3].argmin()
        shrinked_corners = [dc[tl][0,0], dc[tr][0,1], dc[br][0,2], dc[bl][0,3]]

        self.detected_corners = detected_corners
        # avarage_corner_int = np.mean(np.array(detected_corners), axis=0).astype(int)
        
        cv2.destroyAllWindows()

        # add corners to gray

        corners_img_proj = np.array(shrinked_corners).astype(np.float32).reshape(1, 4, 2)
        self.img_non_zero_section = self.img[
            self.orig_rect_corners[0][1]:self.orig_rect_corners[2][1],
            self.orig_rect_corners[0][0]:self.orig_rect_corners[1][0]
        ]

        # print(f'found aruco code {displayed_aruco_code} corners: {corners_img_proj}')
        # add corners_img_proj to gray
        frame_with_corners = cv2.aruco.drawDetectedMarkers(
            frame.copy(), np.array([corners_img_proj]), np.array([displayed_aruco_code])
        )
        # add each corner as dot
        for corner in corners_img_proj[0]:
                corner = corner.astype(int)
                cv2.circle(frame_with_corners, tuple(corner), 5, (255, 0, 0), -1)
        

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

        self.H = cv2.getPerspectiveTransform(corners_img_proj, img_non_zero_section_corners)

        self.frame = frame

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

        self.color_pattern = color_pattern

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
            # clear all plt
            IPython.display.clear_output(wait=True)


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

        photometric_calibrations_dir = './photometric_calibrations'
        cur_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        target_path = os.path.join(photometric_calibrations_dir, f'photometric_calibration_{cur_time}.pkl')
        os.makedirs(photometric_calibrations_dir, exist_ok=True)
        with open(target_path, 'wb') as f:
            pickle.dump({
                'augmentor': augmentor,
                'H': self.H,
                'width': self.width,
                'height': self.height,
                'orig_proj_corners': self.orig_proj_corners,
                'orig_proj_striped_corners': self.orig_proj_striped_corners,
                'orig_rect_corners': self.orig_rect_corners,
            }, f)

    def calibrate_board_perpendicular(self, printed_aruco_id=9):
        """
        Calibrate the system when the board is perpendicular to the projector.
        This captures reference ArUco corners that represent zero rotation.
        
        Args:
            printed_aruco_id: ID of the printed ArUco marker on the board
        
        Returns:
            reference_corners: The detected corners of the printed ArUco when board is perpendicular
        """
        detectorParams = cv2.aruco.DetectorParameters()
        detector = aruco.ArucoDetector(self.aruco_dict, detectorParams)
        
        print(f"Position the board perpendicular to the projector.")
        print(f"Looking for printed ArUco marker ID: {printed_aruco_id}")
        print("Press 'c' to capture reference, 'q' to quit")
        
        reference_corners = None
        
        while reference_corners is None:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            frame_viz = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame_viz, corners, ids)
                
                # Look for the printed ArUco
                idx = np.where(ids.flatten() == printed_aruco_id)[0]
                if len(idx) > 0:
                    printed_corners = corners[idx[0]].reshape(4, 2)
                    for corner in printed_corners:
                        cv2.circle(frame_viz, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
                    cv2.putText(frame_viz, "Marker detected - Press 'c' to capture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_viz, f"Waiting for ArUco #{printed_aruco_id}...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Calibration", frame_viz)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and ids is not None:
                idx = np.where(ids.flatten() == printed_aruco_id)[0]
                if len(idx) > 0:
                    reference_corners = corners[idx[0]].reshape(4, 2)
                    print(f"Reference captured! Corners: {reference_corners}")
                    break
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        self.reference_printed_corners = reference_corners
        self.printed_aruco_id = printed_aruco_id
        
        return reference_corners

    def calibrate_perpendicular_with_projection(self, printed_aruco_id=9):
        """
        Enhanced calibration that captures both the printed ArUco reference
        AND the relationship between a projected square and how it appears in camera.
        This establishes the projector->camera mapping when board is perpendicular.
        
        Args:
            printed_aruco_id: ID of the printed ArUco marker on the board
        
        Returns:
            tuple: (reference_printed_corners, reference_projected_corners_camera)
        """
        # First get the reference printed ArUco position
        reference_corners = self.calibrate_board_perpendicular(printed_aruco_id)
        
        # Now project a square and see where it appears in camera space
        print("\nProjecting reference square to establish projector->camera mapping...")
        
        # Project at original corners
        cv2.namedWindow("Rectangle Window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Rectangle Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Rectangle Window", self.img)
        cv2.imshow("Rectangle Window", self.img)
        cv2.waitKey(1)
        
        import time
        time.sleep(0.5)
        
        # Capture and detect the projected square corners in camera space
        # We'll use the projected ArUco for this
        detectorParams = cv2.aruco.DetectorParameters()
        detector = aruco.ArucoDetector(self.aruco_dict, detectorParams)
        
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Find the projected ArUco (displayed_aruco_code)
        if ids is not None and displayed_aruco_code in ids.flatten():
            idx = np.where(ids.flatten() == displayed_aruco_code)[0]
            reference_projected_corners_camera = corners[idx[0]].reshape(4, 2)
            
            print(f"Reference projection corners in camera space: {reference_projected_corners_camera}")
            
            self.reference_projected_corners_camera = reference_projected_corners_camera
            
            return reference_corners, reference_projected_corners_camera
        else:
            raise ValueError("Could not detect projected ArUco marker. Make sure projection is visible.")

    def compute_rotation_compensated_corners(self, printed_aruco_id=None, visualize=True):
        """
        Compute projection corners that compensate for board rotation.
        
        Correct approach using coordinate system transformations:
        1. Detect how the printed ArUco moved in camera space (board rotation)
        2. Compute how a projected square's corners would need to move in camera space
           to maintain a square shape on the rotated board
        3. Use the existing projector->camera mapping to find projector corners
           that produce the desired camera-space corners
        
        Args:
            printed_aruco_id: ID of the printed ArUco marker (uses self.printed_aruco_id if None)
            visualize: Whether to show visualization of the detection
        
        Returns:
            compensated_corners: New projection corners that will appear square on the rotated board
        """
        if printed_aruco_id is None:
            if not hasattr(self, 'printed_aruco_id'):
                raise ValueError("printed_aruco_id not set. Run calibrate_board_perpendicular() first.")
            printed_aruco_id = self.printed_aruco_id
        
        if not hasattr(self, 'reference_printed_corners'):
            raise ValueError("No reference corners found. Run calibrate_perpendicular_with_projection() first.")
        
        if not hasattr(self, 'reference_projected_corners_camera'):
            raise ValueError("No projection reference. Run calibrate_perpendicular_with_projection() first.")
        
        # Detect current position of printed ArUco
        detectorParams = cv2.aruco.DetectorParameters()
        detector = aruco.ArucoDetector(self.aruco_dict, detectorParams)
        
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to capture frame")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is None or printed_aruco_id not in ids.flatten():
            raise ValueError(f"Printed ArUco marker #{printed_aruco_id} not detected")
        
        # Get current corners of the printed marker in camera space
        idx = np.where(ids.flatten() == printed_aruco_id)[0]
        current_printed_corners = corners[idx[0]].reshape(4, 2)
        
        if visualize:
            frame_viz = frame.copy()
            cv2.aruco.drawDetectedMarkers(frame_viz, corners, ids)
            for i, corner in enumerate(current_printed_corners):
                cv2.circle(frame_viz, tuple(corner.astype(int)), 5, (255, 0, 0), -1)
            for i, corner in enumerate(self.reference_printed_corners):
                cv2.circle(frame_viz, tuple(corner.astype(int)), 7, (0, 255, 0), 2)
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(frame_viz, cv2.COLOR_BGR2RGB))
            plt.title("Current (blue filled) vs Reference (green hollow)")
            plt.show()
        
        # Compute homography: how did the board rotate in camera space?
        H_board_rotation_camera, _ = cv2.findHomography(
            self.reference_printed_corners.astype(np.float32),
            current_printed_corners.astype(np.float32)
        )
        
        # Apply the same rotation to the reference projected corners
        # This tells us where the projection corners SHOULD appear in camera space
        # to maintain the same shape on the rotated board
        ref_proj_homogeneous = np.hstack([
            self.reference_projected_corners_camera,
            np.ones((4, 1))
        ])
        
        target_proj_camera_homogeneous = (H_board_rotation_camera @ ref_proj_homogeneous.T).T
        target_proj_camera = target_proj_camera_homogeneous[:, :2] / target_proj_camera_homogeneous[:, 2:3]
        
        # Now we need to find projector corners that will produce target_proj_camera
        # We use the inverse of the existing projector->camera homography
        # The existing H maps from camera content space to camera view
        # We need projector screen -> camera view mapping
        
        # Create a homography from original projector corners to their camera appearance
        H_proj_to_camera, _ = cv2.findHomography(
            self.orig_proj_corners.astype(np.float32),
            self.reference_projected_corners_camera.astype(np.float32)
        )
        
        # Invert to get camera -> projector mapping
        H_camera_to_proj = np.linalg.inv(H_proj_to_camera)
        
        # Map target camera corners back to projector space
        target_camera_homogeneous = np.hstack([
            target_proj_camera,
            np.ones((4, 1))
        ])
        
        compensated_homogeneous = (H_camera_to_proj @ target_camera_homogeneous.T).T
        compensated_corners = compensated_homogeneous[:, :2] / compensated_homogeneous[:, 2:3]
        
        if visualize:
            print("="*60)
            print("COORDINATE SPACES:")
            print("="*60)
            print("\n1. PROJECTOR SPACE (screen pixels):")
            print("   Original projection corners:", self.orig_proj_corners)
            print("   Compensated corners:", compensated_corners)
            print("\n2. CAMERA SPACE (camera pixels):")
            print("   Reference printed ArUco:", self.reference_printed_corners)
            print("   Current printed ArUco:", current_printed_corners)
            print("   Reference projected corners (perpendicular):", self.reference_projected_corners_camera)
            print("   Target projected corners (rotated):", target_proj_camera)
            print("\n3. TRANSFORMATIONS:")
            print("   Board rotation (camera space):")
            print("   ", H_board_rotation_camera)
            print("="*60)
        
        return compensated_corners.astype(np.float32)
