"""
Spatial Rich Model (SRM) Feature Extraction
===========================================

This module implements efficient SRM feature extraction for steganalysis.
SRM features are computed using high-order residuals from spatial filters
followed by quantization and co-occurrence matrix computation.

Key features:
- GPU acceleration using CuPy when available
- Multiprocessing for batch extraction
- Memory-efficient processing
- Support for various filter types (3x3, 5x5)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Union
import multiprocessing as mp
from functools import partial
import logging
from tqdm import tqdm
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logging.info("CuPy detected - GPU acceleration enabled for SRM extraction")
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available - using CPU for SRM extraction")

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available - some optimizations disabled")


class SRMFeatureExtractor:
    """
    Spatial Rich Model feature extractor with GPU acceleration.
    
    Implements the complete SRM pipeline:
    1. Apply spatial filters to compute residuals
    2. Quantize residuals 
    3. Compute co-occurrence matrices
    4. Extract statistical features
    """
    
    def __init__(self, 
                 filters_3x3: bool = True,
                 filters_5x5: bool = True,
                 quantization_factor: float = 1.0,
                 use_gpu: bool = True,
                 truncation_threshold: int = 2):
        """
        Initialize SRM feature extractor.
        
        Args:
            filters_3x3: Whether to use 3x3 filters
            filters_5x5: Whether to use 5x5 filters  
            quantization_factor: Quantization factor for residuals
            use_gpu: Whether to use GPU acceleration
            truncation_threshold: Threshold for residual truncation
        """
        self.filters_3x3 = filters_3x3
        self.filters_5x5 = filters_5x5
        self.quantization_factor = quantization_factor
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.truncation_threshold = truncation_threshold
        
        # Initialize filters
        self._init_filters()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _init_filters(self):
        """Initialize spatial filters for residual computation."""
        
        # 3x3 filters (first order)
        self.filters_3x3_list = [
            # Horizontal
            np.array([[-1, 2, -1]], dtype=np.float32),
            # Vertical  
            np.array([[-1], [2], [-1]], dtype=np.float32),
            # Diagonal
            np.array([[-1, 2, -1]], dtype=np.float32),
            # Anti-diagonal
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
            # Edge filters
            np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
        ]
        
        # 5x5 filters (higher order)
        self.filters_5x5_list = [
            # Extended horizontal
            np.array([[1, -4, 6, -4, 1]], dtype=np.float32),
            # Extended vertical
            np.array([[1], [-4], [6], [-4], [1]], dtype=np.float32),
            # Complex patterns
            np.array([
                [0, 0, -1, 0, 0],
                [0, -1, 4, -1, 0], 
                [-1, 4, -6, 4, -1],
                [0, -1, 4, -1, 0],
                [0, 0, -1, 0, 0]
            ], dtype=np.float32),
        ]
        
        # Move filters to GPU if available
        if self.use_gpu:
            self.filters_3x3_gpu = [cp.array(f) for f in self.filters_3x3_list]
            self.filters_5x5_gpu = [cp.array(f) for f in self.filters_5x5_list]
    
    def _apply_filter(self, image: np.ndarray, filter_kernel: np.ndarray) -> np.ndarray:
        """Apply spatial filter to image."""
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                # GPU version using CuPy
                img_gpu = cp.array(image, dtype=cp.float32)
                filter_gpu = cp.array(filter_kernel, dtype=cp.float32)
                
                # Use cupyx.scipy.ndimage.convolve instead of convolve2d
                from cupyx.scipy import ndimage
                residual = ndimage.convolve(img_gpu, filter_gpu, mode='constant')
                return cp.asnumpy(residual)
            except ImportError:
                # Fallback to CPU if cupyx.scipy not available
                pass
        
        # CPU version using OpenCV
        return cv2.filter2D(image.astype(np.float32), -1, filter_kernel)
    
    def _quantize_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Quantize residuals with truncation."""
        # Apply quantization
        quantized = np.round(residuals / self.quantization_factor)
        
        # Truncate to threshold
        quantized = np.clip(quantized, -self.truncation_threshold, self.truncation_threshold)
        
        return quantized.astype(np.int8)
    
    def _compute_cooccurrence_matrix(self, residuals: np.ndarray, direction: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """Compute co-occurrence matrix from quantized residuals."""
        
        # Get dimensions
        rows, cols = residuals.shape
        
        # Calculate offset
        dy, dx = direction
        
        # Determine valid region
        if dy >= 0 and dx >= 0:
            valid_rows = slice(0, rows - abs(dy))
            valid_cols = slice(0, cols - abs(dx))
            offset_rows = slice(abs(dy), rows)
            offset_cols = slice(abs(dx), cols)
        else:
            # Handle negative offsets
            valid_rows = slice(abs(dy), rows)
            valid_cols = slice(abs(dx), cols)  
            offset_rows = slice(0, rows - abs(dy))
            offset_cols = slice(0, cols - abs(dx))
        
        # Extract pixel pairs
        pixels1 = residuals[valid_rows, valid_cols].flatten()
        pixels2 = residuals[offset_rows, offset_cols].flatten()
        
        # Shift to positive indices
        min_val = -self.truncation_threshold
        max_val = self.truncation_threshold
        range_size = max_val - min_val + 1
        
        pixels1_shifted = pixels1 - min_val
        pixels2_shifted = pixels2 - min_val
        
        # Compute co-occurrence matrix
        cooc_matrix = np.zeros((range_size, range_size), dtype=np.float32)
        
        if NUMBA_AVAILABLE:
            cooc_matrix = self._compute_cooc_numba(pixels1_shifted, pixels2_shifted, cooc_matrix)
        else:
            # Fallback method
            for i in range(len(pixels1_shifted)):
                cooc_matrix[pixels1_shifted[i], pixels2_shifted[i]] += 1
        
        # Normalize
        total = np.sum(cooc_matrix)
        if total > 0:
            cooc_matrix = cooc_matrix / total
            
        return cooc_matrix
    
    @staticmethod
    @jit(nopython=True, parallel=True) if NUMBA_AVAILABLE else lambda func: func
    def _compute_cooc_numba(pixels1, pixels2, cooc_matrix):
        """Numba-accelerated co-occurrence matrix computation."""
        for i in prange(len(pixels1)):
            cooc_matrix[pixels1[i], pixels2[i]] += 1
        return cooc_matrix
    
    def _extract_texture_features(self, cooc_matrix: np.ndarray) -> np.ndarray:
        """Extract texture features from co-occurrence matrix."""
        
        features = []
        
        # Basic statistics
        features.append(np.sum(cooc_matrix))  # Total
        features.append(np.mean(cooc_matrix))  # Mean
        features.append(np.std(cooc_matrix))   # Standard deviation
        features.append(np.max(cooc_matrix))   # Maximum
        features.append(np.min(cooc_matrix))   # Minimum
        
        # Haralick features
        rows, cols = cooc_matrix.shape
        i, j = np.ogrid[0:rows, 0:cols]
        
        # Energy (Angular Second Moment)
        energy = np.sum(cooc_matrix ** 2)
        features.append(energy)
        
        # Contrast
        contrast = np.sum(cooc_matrix * (i - j) ** 2)
        features.append(contrast)
        
        # Homogeneity (Inverse Difference Moment)
        homogeneity = np.sum(cooc_matrix / (1 + (i - j) ** 2))
        features.append(homogeneity)
        
        # Entropy
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(cooc_matrix * np.log(cooc_matrix + epsilon))
        features.append(entropy)
        
        # Correlation
        mu_i = np.sum(i * cooc_matrix)
        mu_j = np.sum(j * cooc_matrix)
        sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * cooc_matrix))
        sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * cooc_matrix))
        
        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum((i - mu_i) * (j - mu_j) * cooc_matrix) / (sigma_i * sigma_j)
        else:
            correlation = 0
        features.append(correlation)
        
        return np.array(features, dtype=np.float32)
    
    def extract_features_single(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Extract SRM features from a single image.
        
        Args:
            image_input: Path to the image file or numpy array
            
        Returns:
            Feature vector as numpy array
        """
        
        # Handle both file path and numpy array input
        if isinstance(image_input, (str, Path)):
            # Load image in grayscale
            image = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            # Use provided numpy array
            image = image_input
            if len(image.shape) == 3:  # Convert to grayscale if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        all_features = []
        
        # Process 3x3 filters
        if self.filters_3x3:
            for filter_kernel in self.filters_3x3_list:
                # Apply filter
                residuals = self._apply_filter(image, filter_kernel)
                
                # Quantize
                quantized = self._quantize_residuals(residuals)
                
                # Compute co-occurrence matrices in different directions
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal, anti-diagonal
                
                for direction in directions:
                    cooc_matrix = self._compute_cooccurrence_matrix(quantized, direction)
                    texture_features = self._extract_texture_features(cooc_matrix)
                    all_features.extend(texture_features)
        
        # Process 5x5 filters
        if self.filters_5x5:
            for filter_kernel in self.filters_5x5_list:
                # Apply filter
                residuals = self._apply_filter(image, filter_kernel)
                
                # Quantize
                quantized = self._quantize_residuals(residuals)
                
                # Compute co-occurrence matrices
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                
                for direction in directions:
                    cooc_matrix = self._compute_cooccurrence_matrix(quantized, direction)
                    texture_features = self._extract_texture_features(cooc_matrix)
                    all_features.extend(texture_features)
        
        return np.array(all_features, dtype=np.float32)
    
    def extract_features_batch(self, 
                             image_paths: List[Union[str, Path]], 
                             n_jobs: int = -1,
                             batch_size: int = 100) -> np.ndarray:
        """
        Extract SRM features from multiple images using multiprocessing.
        
        Args:
            image_paths: List of image file paths
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            batch_size: Batch size for processing
            
        Returns:
            Feature matrix (n_images, n_features)
        """
        
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        self.logger.info(f"Extracting SRM features from {len(image_paths)} images using {n_jobs} processes")
        
        start_time = time.time()
        
        # Process in batches to manage memory
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            
            with mp.Pool(n_jobs) as pool:
                batch_features = pool.map(self.extract_features_single, batch_paths)
            
            all_features.extend(batch_features)
        
        # Convert to numpy array
        feature_matrix = np.vstack(all_features)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def get_feature_dimension(self) -> int:
        """Get the expected feature dimension."""
        n_features = 0
        
        # Count features from 3x3 filters
        if self.filters_3x3:
            n_features += len(self.filters_3x3_list) * 4 * 10  # filters * directions * texture_features
        
        # Count features from 5x5 filters  
        if self.filters_5x5:
            n_features += len(self.filters_5x5_list) * 4 * 10  # filters * directions * texture_features
            
        return n_features


def extract_srm_features_worker(args):
    """Worker function for multiprocessing feature extraction."""
    image_path, extractor_params = args
    
    # Create extractor instance
    extractor = SRMFeatureExtractor(**extractor_params)
    
    # Extract features
    try:
        features = extractor.extract_features_single(image_path)
        return features
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test SRM feature extraction
    extractor = SRMFeatureExtractor()
    
    print(f"Expected feature dimension: {extractor.get_feature_dimension()}")
    print(f"GPU acceleration: {'Enabled' if extractor.use_gpu else 'Disabled'}")