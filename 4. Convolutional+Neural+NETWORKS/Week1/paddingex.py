import numpy as np
import matplotlib.pyplot as plt

def convolve_with_padding_stride(image, kernel, padding=0, stride=1):
    """
    Performs 2D convolution with specified padding and stride.
    
    Args:
        image (np.array): Input 2D image (H x W)
        kernel (np.array): Filter matrix (f x f)
        padding (int): Number of zero-padding layers
        stride (int): Step size for the filter
        
    Returns:
        output (np.array): Convolved feature map
    """
    # 1. Add Padding
    # Pad input with zeros around the height and width
    image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    # Get dimensions
    (n_H_prev, n_W_prev) = image.shape
    (f, f) = kernel.shape
    
    # 2. Calculate Output Dimensions
    # Formula: floor((n + 2p - f) / s) + 1
    n_H = int((n_H_prev + 2 * padding - f) / stride) + 1
    n_W = int((n_W_prev + 2 * padding - f) / stride) + 1
    
    # Initialize output
    output = np.zeros((n_H, n_W))
    
    # 3. Apply Convolution (Sliding Window)
    for h in range(n_H):            # Loop over vertical axis
        for w in range(n_W):        # Loop over horizontal axis
            # Determine the slice of the padded image
            vert_start = h * stride
            vert_end = vert_start + f
            horiz_start = w * stride
            horiz_end = horiz_start + f
            
            # Extract region of interest
            roi = image_padded[vert_start:vert_end, horiz_start:horiz_end]
            
            # Element-wise multiply and sum
            output[h, w] = np.sum(roi * kernel)
            
    return output

# --- VISUALIZATION SETUP ---

# 1. Create a 7x7 Image with a Vertical Edge (Bright Left, Dark Right)
input_image = np.zeros((7, 7))
input_image[:, :4] = 10  # Left side bright (10)
input_image[:, 4:] = 0   # Right side dark (0)

# 2. Define Vertical Edge Filter (3x3)
vertical_filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# --- EXPERIMENTS ---

# Experiment A: Basic Convolution (No Pad, Stride 1) - "Valid"
# Input: 7x7 -> Output: 5x5 (Shrinks)
output_valid = convolve_with_padding_stride(input_image, vertical_filter, padding=0, stride=1)

# Experiment B: "Same" Convolution (Pad 1, Stride 1)
# Input: 7x7 -> Output: 7x7 (Preserves Size)
output_same = convolve_with_padding_stride(input_image, vertical_filter, padding=1, stride=1)

# Experiment C: Strided Convolution (Pad 0, Stride 2)
# Input: 7x7 -> Output: 3x3 (Downsamples)
output_strided = convolve_with_padding_stride(input_image, vertical_filter, padding=0, stride=2)

# --- PLOTTING ---

fig, ax = plt.subplots(1, 4, figsize=(20, 5))

# Plot Input
ax[0].imshow(input_image, cmap='gray')
ax[0].set_title(f"Input Image (7x7)\nPixel Values: 0-10")

# Plot Valid (Shrinks)
ax[1].imshow(output_valid, cmap='gray')
ax[1].set_title(f"Valid Conv (p=0, s=1)\nOutput: {output_valid.shape}\nNotice: Image Shrank")

# Plot Same (Preserves)
ax[2].imshow(output_same, cmap='gray')
ax[2].set_title(f"Same Conv (p=1, s=1)\nOutput: {output_same.shape}\nNotice: Size Preserved")

# Plot Strided (Shrinks Rapidly)
ax[3].imshow(output_strided, cmap='gray')
ax[3].set_title(f"Strided Conv (p=0, s=2)\nOutput: {output_strided.shape}\nNotice: Downsampling")

plt.tight_layout()
plt.show()
