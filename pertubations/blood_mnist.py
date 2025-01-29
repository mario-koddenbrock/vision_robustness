import cv2
import numpy as np


def perturb_image(image, rotation_angle=0, flip=False, scale_factor=1.0, noise_intensity=0.01,
                  brightness_factor=1.0, contrast_factor=1.0, blur_intensity=0,
                  perspective_shift=(0, 0), focus_variation=0):
    """
    Apply perturbations to an input image, simulating real-world challenges for blood microscopy images.

    Parameters:
    - image: Input image to be perturbed (numpy array)
    - rotation_angle: Degree of random rotation (-180 to 180 degrees)
    - flip: Whether to randomly flip the image horizontally/vertically (True/False)
    - scale_factor: Scaling factor for resizing the image (default is 1.0, no scaling)
    - noise_intensity: Intensity of random noise to add (0 to 1, higher values add more noise)
    - brightness_factor: Factor to adjust brightness (default is 1.0, no change)
    - contrast_factor: Factor to adjust contrast (default is 1.0, no change)
    - blur_intensity: Amount of Gaussian blur to apply (0 for no blur, higher values increase blur)
    - perspective_shift: Tuple indicating perspective shift in x and y directions (default is no shift)
    - focus_variation: Amount of defocus blur to simulate changes in focus (0 for no blur)

    Returns:
    - Perturbed image (numpy array)
    """

    # Make a copy of the image to apply transformations
    perturbed_image = image.copy()

    # 1. Random Rotation
    if rotation_angle != 0:
        (h, w) = perturbed_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        perturbed_image = cv2.warpAffine(perturbed_image, M, (w, h))

    # 2. Random Flip
    if flip:
        flip_type = np.random.choice([-1, 0, 1])  # Randomly choose horizontal, vertical, or both
        perturbed_image = cv2.flip(perturbed_image, flip_type)

    # 3. Scaling
    if scale_factor != 1.0:
        perturbed_image = cv2.resize(perturbed_image, None, fx=scale_factor, fy=scale_factor,
                                     interpolation=cv2.INTER_LINEAR)

    # 4. Adding Gaussian Noise
    if noise_intensity > 0:
        noise = np.random.randn(*perturbed_image.shape) * noise_intensity
        perturbed_image = perturbed_image + noise
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)

    # 5. Brightness and Contrast Adjustment
    perturbed_image = cv2.convertScaleAbs(perturbed_image, alpha=contrast_factor, beta=brightness_factor * 50)

    # 6. Gaussian Blur (to simulate defocus or motion blur)
    if blur_intensity > 0:
        perturbed_image = cv2.GaussianBlur(perturbed_image, (blur_intensity, blur_intensity), 0)

    # 7. Perspective Transform
    if perspective_shift != (0, 0):
        rows, cols = perturbed_image.shape[:2]
        shift_x, shift_y = perspective_shift
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
        pts2 = np.float32([[shift_x, shift_y], [cols + shift_x, shift_y], [shift_x, rows + shift_y]])
        M = cv2.getAffineTransform(pts1, pts2)
        perturbed_image = cv2.warpAffine(perturbed_image, M, (cols, rows))

    # 8. Defocus Blur (to simulate out-of-focus areas in microscopy)
    if focus_variation > 0:
        perturbed_image = cv2.GaussianBlur(perturbed_image, (focus_variation * 2 + 1, focus_variation * 2 + 1),
                                           focus_variation)

    return perturbed_image

# Example usage:
# Assuming 'image' is a numpy array representing your input image
# perturbed = perturb_image(image, rotation_angle=10, flip=True, scale_factor=1.2, noise_intensity=0.05,
#                           brightness_factor=1.1, contrast_factor=1.2, blur_intensity=3,
#                           perspective_shift=(5, 5), focus_variation=2)

