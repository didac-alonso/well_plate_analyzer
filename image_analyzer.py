from skimage import transform
from skimage.io import imread, imshow
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle
import configparser
from tkinter import Tk, filedialog
import csv
import math

# -----------------------------
# Global variable for clicked points
points = []

# -----------------------------
# Point selection functions
def select_points(event, x, y, flags, param):
    """Callback function for selecting points with a mouse click."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < param["num_points"]:
        points.append((x, y))
        print(f"Point selected: {x}, {y}")
        cv2.circle(param["image"], (x, y), 5, (0, 255, 0), -1)  # Draw a small circle on the image
        cv2.imshow(param["window_name"], param["image"])

def load_image():
    """Function to select an image using a file dialog."""
    Tk().withdraw()  # Hides the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        return cv2.imread(file_path), file_path
    else:
        print("No file selected.")
        return None, None

def draw_grid(image, grid_size=20, color=(200, 200, 200), thickness=1):
    """
    Draws a subtle grid overlay on the image to help the user align their selection.
    
    Args:
        image: The input image (numpy array).
        grid_size: Distance in pixels between grid lines.
        color: Color of the grid lines (default is light gray).
        thickness: Thickness of the grid lines.
    
    Returns:
        Image with grid overlay.
    """
    img_with_grid = image.copy()
    h, w = img_with_grid.shape[:2]

    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv2.line(img_with_grid, (x, 0), (x, h), color, thickness)

    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(img_with_grid, (0, y), (w, y), color, thickness)

    return img_with_grid

def select_n_points(image, num_points, window_name="Select Points"):
    """Allows the user to select `num_points` points from an image with a grid overlay."""
    global points
    points = []  # Reset points
    img_with_grid = draw_grid(image, grid_size=50, color=(200, 200, 200), thickness=1)  # Light gray grid

    cv2.imshow(window_name, img_with_grid)
    params = {"image": img_with_grid, "num_points": num_points, "window_name": window_name}
    cv2.setMouseCallback(window_name, select_points, param=params)

    cv2.waitKey(1)  # Allow proper event handling
    while len(points) < num_points:  # Wait until all points are selected
        cv2.waitKey(1)

    cv2.destroyAllWindows()  # Close the window automatically
    if len(points) == num_points:
        return np.array(points, dtype=np.float32)
    else:
        print(f"Error: You must select exactly {num_points} points.")
        return None


# def select_n_points(image, num_points, window_name="Select Points"):
#     """Allows the user to select `num_points` points from an image."""
#     global points
#     points = []  # Reset points
#     img_copy = image.copy()  # Create a copy for display
#     cv2.imshow(window_name, img_copy)
#     params = {"image": img_copy, "num_points": num_points, "window_name": window_name}
#     cv2.setMouseCallback(window_name, select_points, param=params)
#     cv2.waitKey(1)  # Allow for proper event handling
#     while len(points) < num_points:  # Wait until all points are selected
#         cv2.waitKey(1)
#     cv2.destroyAllWindows()  # Close the window automatically
#     if len(points) == num_points:
#         return np.array(points, dtype=np.float32)
#     else:
#         print(f"Error: You must select exactly {num_points} points.")
#         return None

# -----------------------------
# Perspective transformation functions
def order_points(points):
    """
    Orders the points so that they are arranged as:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # top-left
    rect[2] = points[np.argmax(s)]  # bottom-right
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # top-right
    rect[3] = points[np.argmax(diff)]  # bottom-left
    return rect

def get_projection_rectangle(points):
    """
    Generate the projection rectangle (a perfect rectangle) based on 4 points.
    """
    adjusted_points = order_points(points)
    (tl, tr, br, bl) = adjusted_points
    width = int(np.linalg.norm(tr - tl))
    height = int(np.linalg.norm(bl - tl))
    projection = np.array([
        tl,             # top-left
        [tr[0], tl[1]], # top-right
        [tr[0], bl[1]], # bottom-right
        [tl[0], bl[1]]  # bottom-left
    ], dtype="float32")
    print(f"Ordered Points (4): {adjusted_points}")
    print(f"Projection (4): {projection}")
    return (adjusted_points, projection)

def apply_perspective(image, points):
    """
    Apply perspective transformation to map the selected quadrilateral to a rectangle.
    """
    pts, projection = get_projection_rectangle(points)
    matrix = cv2.getPerspectiveTransform(pts, projection)
    h, w = image.shape[:2]
    transformed_image = cv2.warpPerspective(image, matrix, (w, h))
    return transformed_image

# -----------------------------
# Functions for grid computation
def order_three_points(points):
    """
    Orders 3 points as: top-left, top-right, bottom-left.
    Assumes the user selected these from the transformed image.
    """
    triang = np.zeros((3, 2), dtype="float32")
    s = points.sum(axis=1)
    triang[0] = points[np.argmin(s)]  # top-left
    diff = np.diff(points, axis=1)
    triang[1] = points[np.argmin(diff)]  # top-right
    triang[2] = points[np.argmax(diff)]  # bottom-left
    return triang

def select_grid_of_wells(image, selected_points, num_wells_h, num_wells_v):
    """
    Given the 3 selected points (top-left, top-right, bottom-left),
    calculates the grid of well centers by moving only horizontally (x) and vertically (y).
    """
    top_left, top_right, bottom_left = order_three_points(selected_points)
    # Compute full span for the grid based on the selected well
    total_width = (top_right[0] - top_left[0]) * (num_wells_h - 1)
    total_height = (bottom_left[1] - top_left[1]) * (num_wells_v - 1)
    # Compute per-well step sizes (strictly along x and y)
    step_h = total_width / (num_wells_h - 1)
    step_v = total_height / (num_wells_v - 1)
    grid_centers = []
    for i in range(num_wells_v):
        for j in range(num_wells_h):
            center_x = int(top_left[0] + j * step_h)
            center_y = int(top_left[1] + i * step_v)
            grid_centers.append((center_x, center_y))
    # Debug visualization: show grid centers on the image
    debug_image = image.copy()
    for center in grid_centers:
        cv2.circle(debug_image, center, 5, (0, 0, 255), -1)
    cv2.imshow("Well Centers", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("debug_well_centers.jpg", debug_image)
    print("Saved debug image with well centers as 'debug_well_centers.jpg'.")
    print("Calculated Well Centers:")
    for idx, center in enumerate(grid_centers, start=1):
        print(f"Well {idx}: {center}")
    return grid_centers

# -----------------------------
# Region growing functions (watershed-like)
def get_neighbors(coord, shape):
    """
    Returns the 8-connected neighbors of a pixel at 'coord' (x, y) within the image dimensions.
    """
    x, y = coord
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[1] and 0 <= ny < shape[0]:
                neighbors.append((nx, ny))
    return neighbors


# def adaptive_watershed(image, seed, initial_threshold=220, final_threshold=150, threshold_step=10, 
#                        max_iterations=100, max_radius_diff=5, min_growth_per_iter=5):
#     """
#     Improved region-growing function that mimics a watershed algorithm.
#     - Starts growing under a strict threshold
#     - Expands until it slows down significantly
#     - Dynamically relaxes the threshold as needed
#     - Stops if radius change is too large (prevents leaking into other wells)
    
#     Args:
#         image: Grayscale (inverted) image.
#         seed: A tuple (x, y) for the initial well center.
#         initial_threshold: Starting pixel intensity for inclusion.
#         final_threshold: Lowest pixel intensity allowed for expansion.
#         threshold_step: Amount by which the threshold is relaxed.
#         max_iterations: Max iterations to prevent infinite growth.
#         max_radius_diff: Max allowed change in radius per iteration.
#         min_growth_per_iter: Minimum number of pixels added per iteration.
    
#     Returns:
#         region: Set of pixels in the well.
#         iterations: How many iterations were performed.
#         area: Total number of pixels in the well.
#     """
#     region = set([seed])
#     border = set([seed])
#     iterations = 0
#     threshold = initial_threshold
#     prev_radius = 1  # Start with a tiny circle assumption

#     while iterations < max_iterations and threshold >= final_threshold:
#         new_border = set()

#         for pixel in border:
#             for neighbor in get_neighbors(pixel, image.shape):
#                 if neighbor not in region:
#                     if image[neighbor[1], neighbor[0]] >= threshold:
#                         new_border.add(neighbor)

#         if len(new_border) < min_growth_per_iter:  # Growth has slowed too much
#             threshold -= threshold_step  # Relax threshold
#             continue  # Retry with relaxed threshold

#         new_region = region.union(new_border)
#         new_area = len(new_region)
#         new_radius = math.sqrt(new_area / math.pi)  # Estimate circular radius

#         # If the radius change is too large, assume it leaked into another well
#         if abs(new_radius - prev_radius) > max_radius_diff:
#             print(f"Stopping: Radius grew too fast at iteration {iterations} ({prev_radius:.2f} → {new_radius:.2f})")
#             return set(), iterations, 0  # Invalid region

#         prev_radius = new_radius
#         region = new_region
#         border = new_border
#         iterations += 1

#     area = len(region)
#     return region, iterations, area


# def adaptive_watershed(image, seed, final_threshold=85, threshold_step=10, 
#                        max_iterations=100, max_radius_diff=10, min_growth_per_iter=5):
#     """
#     Adaptive region-growing function for INVERTED images:
#     - Starts at the darkest point (seed intensity).
#     - Expands outward until reaching a threshold of ~175.
#     - Stops if growth slows down or leaks into other wells.

#     Args:
#         image: Grayscale (inverted) image.
#         seed: A tuple (x, y) for the initial well center.
#         final_threshold: The max allowed threshold for expansion.
#         threshold_step: How much to relax the threshold per iteration.
#         max_iterations: Max iterations to prevent infinite growth.
#         max_radius_diff: Max allowed change in radius per iteration.
#         min_growth_per_iter: Minimum number of pixels added per iteration.

#     Returns:
#         region: Set of pixels in the well.
#         iterations: Number of iterations performed.
#         area: Total number of pixels in the well.
#     """
#     seed_intensity = image[seed[1], seed[0]]  # Get seed pixel intensity
#     threshold = seed_intensity  # Start at the darkest pixel
#     region = set([seed])
#     border = set([seed])
#     iterations = 0
#     prev_radius = 1  # Initial small circle assumption

#     while iterations < max_iterations and threshold <= final_threshold:
#         new_border = set()

#         for pixel in border:
#             for neighbor in get_neighbors(pixel, image.shape):
#                 if neighbor not in region:
#                     if image[neighbor[1], neighbor[0]] <= threshold:  # Grow into brighter areas
#                         new_border.add(neighbor)

#         if len(new_border) < min_growth_per_iter:  # If growth slows, increase threshold
#             threshold += threshold_step
#             continue  

#         new_region = region.union(new_border)
#         new_area = len(new_region)
#         new_radius = math.sqrt(new_area / math.pi)  

#         # If the radius change is too large, assume it leaked into another well
#         if abs(new_radius - prev_radius) > max_radius_diff:
#             print(f"Stopping: Radius grew too fast at iteration {iterations} ({prev_radius:.2f} → {new_radius:.2f})")
#             return set(), iterations, 0  

#         prev_radius = new_radius
#         region = new_region
#         border = new_border
#         iterations += 1

#     area = len(region)
#     return region, iterations, area


import numpy as np
import math

def adaptive_watershed(image, seed, initial_threshold=150, final_threshold=190, threshold_step=10, 
                       max_iterations=200, max_radius_diff=5, min_growth_per_iter=5, 
                       max_intensity_jump=20):
    """
    Adaptive region-growing function for INVERTED images:
    - Starts at the darkest point (seed intensity).
    - Expands outward until reaching a threshold.
    - Stops if growth slows down or leaks into other wells.
    - NEW: Stops if there's a sudden intensity jump relative to the seed intensity.

    Args:
        image: Grayscale (inverted) image.
        seed: A tuple (x, y) for the initial well center.
        initial_threshold: The max initial seed intensity allowed.
        final_threshold: The max allowed threshold for expansion.
        threshold_step: How much to relax the threshold per iteration.
        max_iterations: Max iterations to prevent infinite growth.
        max_radius_diff: Max allowed change in radius per iteration.
        min_growth_per_iter: Minimum number of pixels added per iteration.
        max_intensity_jump: Maximum allowed jump in intensity relative to the seed.

    Returns:
        region: Set of pixels in the well.
        iterations: Number of iterations performed.
        area: Total number of pixels in the well.
    """

    seed_intensity = image[seed[1], seed[0]]  # Get seed pixel intensity
    threshold = seed_intensity  # Start at the darkest pixel

    region = set([seed])
    border = set([seed])
    iterations = 0
    prev_radius = 1  # Initial small circle assumption
    
    if seed_intensity > initial_threshold:
        return set(), 0, 0  # If the seed is too bright, discard it

    while iterations < max_iterations and threshold <= final_threshold:
        new_border = set()
        new_max_intensity = 0  # Track max intensity in new region

        for pixel in border:
            for neighbor in get_neighbors(pixel, image.shape):
                if neighbor not in region:
                    pixel_intensity = image[neighbor[1], neighbor[0]]
                    if pixel_intensity <= threshold:  # Grow into brighter areas
                        new_border.add(neighbor)
                        new_max_intensity = max(new_max_intensity, pixel_intensity)

        if len(new_border) < min_growth_per_iter:
            threshold += threshold_step
            continue

        # Check for sudden intensity jump relative to the seed intensity
        if new_max_intensity - seed_intensity > max_intensity_jump & new_max_intensity > seed_intensity:
            print(f"Stopping: Intensity jumped from seed {seed_intensity} → {new_max_intensity} at iteration {iterations}")
            return region, iterations, len(region)  # Return detected region and area

        # Compute new radius and check for well leakage
        new_region = region.union(new_border)
        new_area = len(new_region)
        new_radius = math.sqrt(new_area / math.pi)  

        if abs(new_radius - prev_radius) > max_radius_diff:
            print(f"Stopping: Radius grew too fast at iteration {iterations} ({prev_radius:.2f} → {new_radius:.2f})")
            return region, iterations, len(region)  # Return detected region and area

        # Update values for next iteration
        prev_radius = new_radius
        region = new_region
        border = new_border
        iterations += 1

    return region, iterations, len(region)  # Return final region and area

def region_growing(image, seed, pixel_threshold=200, max_iterations=50, max_radius_diff=5):
    """
    Perform iterative region-growing segmentation from a seed point on a grayscale (inverted) image.
    The algorithm stops either when no new pixels are added, when the maximum iterations is reached,
    or when the difference in estimated radius between two iterations exceeds max_radius_diff,
    indicating that the region has "escaped" the well.
    
    Args:
      image: Grayscale (inverted) image.
      seed: A tuple (x, y) representing the seed coordinate.
      pixel_threshold: The intensity threshold for including a pixel (default 200).
      max_iterations: Maximum number of iterations (default 50).
      max_radius_diff: Maximum allowed increase in estimated radius between iterations (default 5 pixels).
      
    Returns:
      region: A set of pixel coordinates in the grown region (empty if region escaped).
      iterations: Number of iterations performed.
      area: Total number of pixels in the region (0 if region escaped).
    """
    region = set([seed])
    border = set([seed])
    iterations = 0
    # Start with an estimated radius corresponding to one pixel.
    prev_radius = math.sqrt(1 / math.pi)
    while iterations < max_iterations:
        new_border = set()
        for pixel in border:
            for neighbor in get_neighbors(pixel, image.shape):
                if neighbor not in region:
                    # Note: image is indexed as image[y, x]
                    if image[neighbor[1], neighbor[0]] >= pixel_threshold:
                        new_border.add(neighbor)
        if not new_border:
            break
        new_region = region.union(new_border)
        new_area = len(new_region)
        new_radius = math.sqrt(new_area / math.pi)
        # If the increase in radius is too big, assume region has escaped the well.
        if iterations > 0 and (new_radius - prev_radius) > max_radius_diff:
            print(f"Region growth too large at iteration {iterations}: new radius {new_radius:.2f}, previous {prev_radius:.2f}")
            return set(), iterations, 0
        prev_radius = new_radius
        region = new_region
        border = new_border
        iterations += 1
    area = len(region)
    return region, iterations, area

def classify_and_filter_wells(well_results):
    """
    Classifies well residues based on area and whiteness using dynamic quantiles.
    Filters out wells that don't meet the criteria (area < 50 or whiteness < 100).
    Class 1 = highest whiteness (most white), Class 5 = lowest valid whiteness.
    
    Args:
        well_results: List of tuples [(center, iterations, area, circle_diameter, avg_whiteness), ...]
    
    Returns:
        List of tuples [(center, iterations, updated_area, updated_diameter, avg_whiteness, whiteness_class), ...]
    """
    # First pass: filter wells and collect valid whiteness values
    valid_whiteness_values = []
    
    for center, iter_count, area, circle_diameter, avg_whiteness in well_results:
        # Filter wells based on area and whiteness criteria
        if circle_diameter >= 20 and avg_whiteness >= 120:
            valid_whiteness_values.append(avg_whiteness)
    
    # If no valid wells, return early with all classified as -1
    if not valid_whiteness_values:
        filtered_results = [(center, iter_count, area, circle_diameter, avg_whiteness, -1) 
                           for center, iter_count, area, circle_diameter, avg_whiteness in well_results]
        return filtered_results
    
    # Calculate dynamic quantile thresholds (5 classes)
    valid_whiteness_values.sort()
    num_vals = len(valid_whiteness_values)
    
    # Calculate quantile thresholds (20%, 40%, 60%, 80%)
    thresholds = [
        valid_whiteness_values[int(num_vals * 0.2)],
        valid_whiteness_values[int(num_vals * 0.4)],
        valid_whiteness_values[int(num_vals * 0.6)],
        valid_whiteness_values[int(num_vals * 0.8)]
    ]
    
    print(f"Dynamic classification thresholds: {thresholds}")
    
    # Second pass: classify wells based on dynamic thresholds
    filtered_results = []
    
    for center, iter_count, area, circle_diameter, avg_whiteness in well_results:
        # Default classification
        whiteness_class = -1
        updated_area = area
        updated_diameter = circle_diameter
        
        # Filter wells based on area and whiteness criteria
        if circle_diameter < 20 or avg_whiteness < 120:
            whiteness_class = -1
            updated_area = 0  # Set area to 0 if filtering out the well
            updated_diameter = 0  # Also update diameter
        else:
            # Dynamic Classification based on quantiles
            # Class 1 = highest whiteness (reversed from previous version)
            if avg_whiteness >= thresholds[3]:
                whiteness_class = 1  # Top 20% (most white)
            elif avg_whiteness >= thresholds[2]:
                whiteness_class = 2  # 60-80%
            elif avg_whiteness >= thresholds[1]:
                whiteness_class = 3  # 40-60%
            elif avg_whiteness >= thresholds[0]:
                whiteness_class = 4  # 20-40%
            else:
                whiteness_class = 5  # Bottom 20% (least white but still valid)
        
        # Append the updated result
        filtered_results.append((center, iter_count, updated_area, updated_diameter, avg_whiteness, whiteness_class))
    
    return filtered_results


def preprocess_image(image):
    """
    Aplica preprocesamiento a la imagen: escala de grises, filtro Gaussiano, CLAHE e inversión.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Mejora de contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)

    # Invertimos la imagen para resaltar los residuos
    inverted = cv2.bitwise_not(enhanced)

    return inverted

# -----------------------------
# Detección de residuos en los pocillos
def detect_residues(image, centers, min_area=50, max_area=2000):
    """
    Detecta residuos en cada pocillo basado en contornos internos.
    """
    well_residues = []
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for (x, y) in centers:
        # Encontrar el contorno más cercano al centro del pocillo
        closest_contour = min(contours, key=lambda c: cv2.pointPolygonTest(c, (x, y), True))

        # Filtrar por tamaño
        area = cv2.contourArea(closest_contour)
        if area < min_area or area > max_area:
            well_residues.append((0, 0, 0))
            continue

        # Ajustar una elipse si hay suficientes puntos
        if len(closest_contour) >= 5:
            ellipse = cv2.fitEllipse(closest_contour)
            major_axis = max(ellipse[1]) / 2
            minor_axis = min(ellipse[1]) / 2
            equivalent_radius = np.sqrt((major_axis * minor_axis) / np.pi)

            well_residues.append((major_axis, minor_axis, equivalent_radius))
            
            # Dibujar el contorno y la elipse en la imagen original
            cv2.drawContours(image, [closest_contour], -1, (0, 255, 0), 2)
            cv2.ellipse(image, ellipse, (0, 0, 255), 2)


    cv2.imwrite('debug_contours.jpg', image)
    
    return well_residues

# -----------------------------
# Main program flow
if __name__ == "__main__":
    # Load image
    image, file_path = load_image()
    if image is None:
        print("Exiting program.")
        exit(0)
    print(f"Loaded image: {file_path}")
    
    # Convert to grayscale and apply binary threshold
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 227, 255, cv2.THRESH_BINARY)
    # Inpaint the image
    image_inpainted = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # Step 1: Select 4 points for perspective transformation
    print("Select 4 points for perspective transformation.")
    points_4 = select_n_points(
        image_inpainted, 4,
        "Select the limits of the plaque to change the perspective (4 points) "
        "try to align with the limits of the last well for each corner"
    )
    if points_4 is None:
        print("Perspective transformation not applied due to insufficient points.")
        exit(0)
    print(f"Selected Points (4): {points_4}")
    
    # Apply perspective transformation
    transformed_image = apply_perspective(image_inpainted, points_4)
    
    # Step 2: Select 3 points in the transformed image
    print("Select 3 points in the transformed image.")
    points_3 = select_n_points(
        transformed_image, 3,
        "Select the centers of the top left well, the one next to it, and the one under"
    )
    if points_3 is None:
        print("Error: You must select exactly 3 points.")
        exit(0)
    print(f"Selected Points (3): {points_3}")
    
    # Ask for grid dimensions (number of wells horizontally and vertically)
    num_wells_h = int(input("Enter the number of wells horizontally: "))
    num_wells_v = int(input("Enter the number of wells vertically: "))
    
    # Compute the grid of well centers from the 3 selected points
    well_centers = select_grid_of_wells(transformed_image, points_3, num_wells_h, num_wells_v)
    
    # Convert the transformed image to grayscale and invert it
    gray_transformed = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_transformed)
    cv2.imwrite("debug_inverted_image.jpg", inverted_image)

    
    cv2.imshow("Inverted Image", inverted_image)
    cv2.waitKey(1000)
    cv2.destroyWindow("Inverted Image")
    
    # Parameters for region growing and validity checking
    pixel_threshold = 200  # Adjust as needed for your image
    max_iterations = 100   # Maximum iterations to prevent flooding
    max_diameter = 224      # Maximum allowed diameter (in pixels) for a valid well region
    # max_radius_diff = 100    # Maximum allowed increase in radius per iteration
    
    # For each well center, perform region growing and record measurements.
    # We will compute:
    #   - Iterations (how many iterations of growth occurred)
    #   - Area (number of pixels in the region)
    #   - "Circle diameter" (approximated from the area as if the region were a circle)
    #   - Average whiteness (computed from the re-inverted pixel values)
    
    # # Ahora aplicamos preprocesamiento después de definir los centros
    # preprocessed_image = preprocess_image(transformed_image)
    # cv2.imwrite("debug_processed_image.jpg", preprocessed_image)

    # # Detectar residuos en los pocillos
    # residue_sizes = detect_residues(preprocessed_image, well_centers)

    # # Ensure detected residues match grid dimensions
    # if len(residue_sizes) != num_wells_h * num_wells_v:
    #     print("Warning: The number of well results does not match the grid dimensions!")

    # # Initialize empty 2D grids
    # major_axis_grid = []
    # minor_axis_grid = []
    # eq_radius_grid = []

    # for i in range(num_wells_v):
    #     row_major = []
    #     row_minor = []
    #     row_eq_radius = []
        
    #     for j in range(num_wells_h):
    #         idx = i * num_wells_h + j
    #         if idx < len(residue_sizes):
    #             major, minor, eq_radius = residue_sizes[idx]
    #         else:
    #             major, minor, eq_radius = 0, 0, 0  # Default if missing values

    #         row_major.append(major)
    #         row_minor.append(minor)
    #         row_eq_radius.append(eq_radius)
        
    #     major_axis_grid.append(row_major)
    #     minor_axis_grid.append(row_minor)
    #     eq_radius_grid.append(row_eq_radius)

    # # Save the grids to CSV
    # def save_csv(filename, grid):
    #     with open(filename, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerows(grid)

    # save_csv("well_major_axis.csv", major_axis_grid)
    # save_csv("well_minor_axis.csv", minor_axis_grid)
    # save_csv("well_eq_radius.csv", eq_radius_grid)

    # print("Saved 'well_major_axis.csv', 'well_minor_axis.csv', and 'well_eq_radius.csv'.")

    # # Guardar resultados en CSV
    # with open("residue_sizes.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["X", "Y", "Major Axis", "Minor Axis", "Equivalent Radius"])
    #     for (x, y), (major, minor, eq_radius) in zip(well_centers, residue_sizes):
    #         writer.writerow([x, y, major, minor, eq_radius])

    # print("Residue sizes saved to 'residue_sizes.csv'.")
    
    well_results = []  # List of tuples: (center, iterations, area, circle_diameter, avg_whiteness)
    for center in well_centers:
        region, iter_count, area = adaptive_watershed(inverted_image, center)

        if area > 0:
            circle_diameter = 2 * math.sqrt(area / math.pi)  # Convert area to diameter
            avg_whiteness = np.mean([255 - inverted_image[pt[1], pt[0]] for pt in region])
        else:
            circle_diameter = 0
            avg_whiteness = 0

        # Apply max diameter rule
        if circle_diameter > max_diameter:
            print(f"Well at {center} escaped! (Diameter {circle_diameter:.2f} > max {max_diameter}).")
            circle_diameter = 0
            avg_whiteness = 0

        well_results.append((center, iter_count, area, circle_diameter, avg_whiteness))
        print(f"Center {center}: Iterations = {iter_count}, Area = {area}, "
            f"Circle Diameter = {circle_diameter:.2f}, Avg Whiteness = {avg_whiteness:.2f}")
        
        
    # for center in well_centers:
    #     region, iter_count, area = region_growing(inverted_image, center, pixel_threshold, max_iterations, max_radius_diff)
    #     if area > 0:
    #         circle_diameter = 2 * math.sqrt(area / math.pi)
    #         # Re-invert the pixel values to compute the original "whiteness"
    #         avg_whiteness = np.mean([255 - inverted_image[pt[1], pt[0]] for pt in region])
    #     else:
    #         circle_diameter = 0
    #         avg_whiteness = 0
    #     # If the computed diameter exceeds max_diameter, mark as invalid (escaped well)
    #     if circle_diameter > max_diameter:
    #         print(f"Well at {center} escaped (diameter {circle_diameter:.2f} > max {max_diameter}).")
    #         circle_diameter = 0
    #         avg_whiteness = 0
    #     well_results.append((center, iter_count, area, circle_diameter, avg_whiteness))
    #     print(f"Center {center}: Iterations = {iter_count}, Area = {area}, "
    #           f"Circle Diameter = {circle_diameter:.2f}, Avg Whiteness = {avg_whiteness:.2f}")
    

    # Apply classification and filtering to well results
    classified_results = classify_and_filter_wells(well_results)

    # Build three grids (2D lists) from classified_results in row-major order
    if len(classified_results) != num_wells_h * num_wells_v:
        print("Warning: The number of well results does not match the grid dimensions!")

    white_intensity_grid = []
    size_grid = []
    class_grid = []  # New grid for classifications

    for i in range(num_wells_v):
        row_white = []
        row_size = []
        row_class = []  # New row for classifications
        
        for j in range(num_wells_h):
            idx = i * num_wells_h + j
            if idx < len(classified_results):
                row_white.append(classified_results[idx][4])  # Average whiteness
                row_size.append(classified_results[idx][3])   # Updated circle diameter (size)
                row_class.append(classified_results[idx][5])  # Whiteness classification
            else:
                # Handle case where there are fewer results than expected
                row_white.append(0)
                row_size.append(0)
                row_class.append(-1)
                
        white_intensity_grid.append(row_white)
        size_grid.append(row_size)
        class_grid.append(row_class)

    # Save the white intensity grid to CSV
    with open("well_white_intensity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(white_intensity_grid)
    print("Saved 'well_white_intensity.csv'.")

    # Save the size grid to CSV
    with open("well_size.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(size_grid)
    print("Saved 'well_size.csv'.")

    # Save the classification grid to CSV
    with open("well_classification.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(class_grid)
    print("Saved 'well_classification.csv'.")

    # Calculate the size-class product grid from the existing grids
    product_grid = []
    for i in range(num_wells_v):
        row_product = []
        for j in range(num_wells_h):
            # If class is -1 (invalid well), product is 0
            if class_grid[i][j] == -1:
                product = 0
            else:
                product = size_grid[i][j] * class_grid[i][j]
            row_product.append(product)
        product_grid.append(row_product)

    # Save the size-class product grid to CSV
    with open("well_size_class_product.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(product_grid)
    print("Saved 'well_size_class_product.csv'.")

    # Build two grids (2D lists) from well_results in row-major order.
    # if len(well_results) != num_wells_h * num_wells_v:
    #     print("Warning: The number of well results does not match the grid dimensions!")


    
    # white_intensity_grid = []
    # size_grid = []
    # for i in range(num_wells_v):
    #     row_white = []
    #     row_size = []
    #     for j in range(num_wells_h):
    #         idx = i * num_wells_h + j
    #         row_white.append(well_results[idx][4])  # Average whiteness
    #         row_size.append(well_results[idx][3])     # Circle diameter (size)
    #     white_intensity_grid.append(row_white)
    #     size_grid.append(row_size)
    
    # # Save the white intensity grid to CSV
    # with open("well_white_intensity.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(white_intensity_grid)
    # print("Saved 'well_white_intensity.csv'.")
    
    # # Save the size grid to CSV
    # with open("well_size.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(size_grid)
    # print("Saved 'well_size.csv'.")