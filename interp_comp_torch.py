import torch
import numpy as np

class UltraOptimizedProjectorCompensation5(torch.nn.Module):
    """
    Projector compensation module.

    Inputs now:
    - C: Captured images array (N, H, W, 3) in (preferably) linear RGB.
    - P: Projected patterns array (N, H, W, 3) in linear RGB (projector input values 0..1).
    - anchors_gray / projected_anchors_gray: For subsequent per-channel interpolation.

    We estimate a per-pixel 3x3 color mixing matrix V via least squares (like estimate_V):
        For each pixel, solve  min_V || P_pix @ V - C_pix ||^2 + lam * ||V - I||^2
        Closed form: V = (X^T X + lam I)^{-1} X^T Y
    where X (N,3), Y (N,3).
    """

    def __init__(
        self,
        C,
        P,
        anchors_gray,
        projected_anchors_gray,
        C_black=None,
        lam: float = 0.0,
        assume_sorted: bool = True,
        device ='cuda'
    ):
        super().__init__()

        self.device = device

        # Ensure tensors & float32
        if not isinstance(C, torch.Tensor):
            C = torch.from_numpy(np.asarray(C)).float()
        if not isinstance(P, torch.Tensor):
            P = torch.from_numpy(np.asarray(P)).float()
        if C_black is not None and not isinstance(C_black, torch.Tensor):
            C_black = torch.from_numpy(np.asarray(C_black)).float()

        # Expect shape (N,H,W,3)
        assert C.ndim == 4 and P.ndim == 4 and C.shape == P.shape, "C and P must be (N,H,W,3) same shape"

        N, H, W, _ = C.shape
        self.H, self.W = H, W
        self.n_samples = len(anchors_gray)

        C = C.to(self.device)
        P = P.to(self.device)
        if C_black is not None:
            C_black = C_black.to(self.device)

        # Baseline subtraction if provided
        if C_black is not None:
            C = C - C_black.unsqueeze(0)  # broadcast if (H,W,3)

        # Estimate per-pixel color mixing matrix V (H,W,3,3)
        self.V = self._estimate_color_mixing_matrix(P, C, lam=lam)

        # Pre-compute interpolation data
        self.x_data_tensor, self.y_data_tensor = self._prepare_interpolation_data(
            anchors_gray, projected_anchors_gray, assume_sorted=assume_sorted
        )

        # Indices grid
        self.h_idx, self.w_idx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )

    def _estimate_color_mixing_matrix(self, P: torch.Tensor, C: torch.Tensor, lam: float = 0.0):
        """Vectorized per-pixel least squares to estimate V s.t. C ≈ P @ V.

        Args:
            P: (N,H,W,3) projector inputs
            C: (N,H,W,3) captured outputs
            lam: ridge regularization (pull towards identity)
        Returns:
            V: (H,W,3,3)
        """
        # Move channels last -> we already have (N,H,W,3)
        N, H, W, _ = P.shape

        # Reshape to (H,W,N,3) for convenience
        X = P.permute(1, 2, 0, 3)  # (H,W,N,3)
        Y = C.permute(1, 2, 0, 3)  # (H,W,N,3)

        # Compute XtX and XtY
        # XtX: sum_n X_n^T X_n -> (H,W,3,3)
        XtX = torch.einsum('hwni,hwnj->hwij', X, X)
        XtY = torch.einsum('hwni,hwnj->hwij', X, Y)  # (H,W,3,3) where columns correspond to output channels

        # Regularization towards identity
        if lam > 0.0:
            eye = torch.eye(3, device=X.device).view(1, 1, 3, 3)
            XtX = XtX + lam * eye
            # Optional pull towards identity in numerator: XtY + lam * I (common in some formulations)
            XtY = XtY + lam * eye

        # Solve (XtX) V = XtY  -> V = (XtX)^{-1} XtY
        # Use torch.linalg.solve for stability
        # Flatten pixels to batch
        XtX_flat = XtX.view(-1, 3, 3)
        XtY_flat = XtY.view(-1, 3, 3)
        V_flat = torch.linalg.solve(XtX_flat, XtY_flat)  # (H*W,3,3)
        V = V_flat.view(H, W, 3, 3)
        return V
    
    def _prepare_interpolation_data(self, anchors_gray, projected_anchors_gray, assume_sorted=True):
        """Pre-compute and organize interpolation data for vectorized processing"""
        
        # Convert to proper format and ensure on GPU
        # anchors_gray: (n_samples, 3, H, W) -> (3, H, W, n_samples)
        x_data_tensor = anchors_gray.permute(1, 2, 3, 0).to(self.device)
        
        # projected_anchors_gray:  (n_samples, 3, H, W) -> (3, H, W, n_samples)
        y_data_tensor = projected_anchors_gray.permute(1, 2, 3, 0).to(self.device)

        if assume_sorted:
            # If we assume the data is already sorted, we can skip the sorting step
            sorted_x_data = x_data_tensor
            sorted_y_data = y_data_tensor
        else:
            # Sort x data and corresponding y data for each pixel and channel
            sorted_x_data = torch.zeros_like(x_data_tensor)
            sorted_y_data = torch.zeros_like(y_data_tensor)

            for c in range(3):
                for h in range(self.H):
                    for w in range(self.W):
                        # Sort indices for this pixel and channel
                        sort_indices = torch.argsort(x_data_tensor[c, h, w])
                        sorted_x_data[c, h, w] = x_data_tensor[c, h, w, sort_indices]
                        sorted_y_data[c, h, w] = y_data_tensor[c, h, w, sort_indices]
            

        return sorted_x_data, sorted_y_data
    
    def forward(self, input_image):
        """
        Ultra-optimized forward pass for projector compensation
        
        Args:
            input_image: (3, H, W) - Input image to compensate
            
        Returns:
            compensated_image: (3, H, W) - Compensated output image
        """
        # Ensure input is on correct device and dtype
        if input_image.device != torch.device(self.device):
            input_image = input_image.to(self.device)
        if input_image.dtype != torch.float32:
            input_image = input_image.float()
        
        # Step 1: Apply color mixing matrix transformation
        # Reshape for vectorized matrix multiplication
        proj_tex_flat = input_image.view(3, -1).permute(1, 0)  # (H*W, 3)
        V_flat = self.V.view(-1, 3, 3)  # (H*W, 3, 3)
        
        # Vectorized matrix multiplication: (H*W, 1, 3) @ (H*W, 3, 3) -> (H*W, 3)
        p_v_flat = torch.bmm(proj_tex_flat.unsqueeze(1), V_flat).squeeze(1)
        
        # Reshape for interpolation: (H*W, 3) -> (3, H, W)
        p_v_reshaped = p_v_flat.view(self.H, self.W, 3).permute(2, 0, 1)
        
        # Step 2: Ultra-fast vectorized interpolation
        compensated_image = torch.zeros_like(p_v_reshaped)
        
        for c in range(3):
            # Get interpolation data for this channel
            x_vals = self.x_data_tensor[c]  # (H, W, n_samples)
            y_vals = self.y_data_tensor[c]  # (H, W, n_samples)
            xi = p_v_reshaped[c]            # (H, W)
            
            # Vectorized searchsorted for all pixels simultaneously
            xi_expanded = xi.unsqueeze(-1)  # (H, W, 1)
            indices = torch.searchsorted(x_vals, xi_expanded, right=False)
            indices = torch.clamp(indices, 1, self.n_samples - 1).squeeze(-1)
            
            # Vectorized linear interpolation using advanced indexing
            x0 = x_vals[self.h_idx, self.w_idx, indices - 1]
            x1 = x_vals[self.h_idx, self.w_idx, indices]
            y0 = y_vals[self.h_idx, self.w_idx, indices - 1]
            y1 = y_vals[self.h_idx, self.w_idx, indices]
            
            # Linear interpolation: y = y0 + α(y1 - y0)
            alpha = (xi - x0) / (x1 - x0 + 1e-8)
            compensated_image[c] = y0 + alpha * (y1 - y0)
        
        return compensated_image
    
    def process_batch(self, batch_images):
        """
        Process a batch of images efficiently
        
        Args:
            batch_images: (batch_size, 3, H, W) - Batch of input images
            
        Returns:
            compensated_batch: (batch_size, 3, H, W) - Batch of compensated images
        """
        batch_size = batch_images.shape[0]
        compensated_batch = torch.zeros_like(batch_images)
        
        for i in range(batch_size):
            compensated_batch[i] = self.forward(batch_images[i])
        
        return compensated_batch
    
    def get_statistics(self):
        """Get statistics about the compensation model"""
        return {
            'image_size': (self.H, self.W),
            'n_samples': self.n_samples,
            'v_matrix_range': {
                'min': float(self.V.min().item()),
                'max': float(self.V.max().item()),
                'mean': float(self.V.mean().item())
            },
            'interpolation_data_range': {
                'x_min': self.x_data_tensor.min().item(),
                'x_max': self.x_data_tensor.max().item(),
                'y_min': self.y_data_tensor.min().item(),
                'y_max': self.y_data_tensor.max().item()
            }
        }
