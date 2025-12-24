import cv2
import torch

border_size = 2

displayed_aruco_code = 42

marker_size = 5000  # Size in pixels

aruco_dict_type = cv2.aruco.DICT_4X4_50  # ArUco dictionary type

latent_size = 16  # Spatial size of the latent (latent shape will be [batch, 4, latent_size, latent_size])

latent_batch_size = 25  # Number of patches in the latent batch

orig_clases = torch.tensor([817, 705, 609, 586, 436, 627, 468, 621, 803, 407, 408, 751, 717,866, 661, 864]).cuda()
