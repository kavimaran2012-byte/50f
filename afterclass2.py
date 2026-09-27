import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """Loads an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"\n[Error] Could not find or open image at: '{image_path}'")
        return None
    return image

def display_side_by_side(original, processed, title="Processed Image"):
    """Displays the original and processed image side by side using Matplotlib."""
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Processed Image
    plt.subplot(1, 2, 2)
    if len(processed.shape) == 2:  # Grayscale / Edge detection output
        plt.imshow(processed, cmap="gray")
    else:  # Color image output
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def apply_edge_detection(image, method="sobel", ksize=3, threshold1=100, threshold2=200):
    """Applies the selected edge detection method to the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "sobel":
        # Ensure ksize is odd and >= 1
        if ksize % 2 == 0:
            ksize += 1
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
        # Combine magnitude of gradients
        magnitude = cv2.magnitude(sobelx, sobely)
        return cv2.convertScaleAbs(magnitude)
        
    elif method == "canny":
        return cv2.Canny(gray_image, threshold1, threshold2)
        
    elif method == "laplacian":
        lap = cv2.Laplacian(gray_image, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)

def apply_filter(image, filter_type="gaussian", ksize=5):
    """Applies Gaussian or Median filtering to smooth out noise."""
    # Kernel size must be an odd number
    if ksize % 2 == 0:
        ksize += 1
        print(f"Kernel size adjusted to next odd integer: {ksize}")
        
    if filter_type == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif filter_type == "median":
        return cv2.medianBlur(image, ksize)

def save_image(image):
    """Saves the processed image to disk with a custom filename."""
    if image is None:
        print("\n[Warning] No processed image available to save!")
        return
        
    filename = input("\nEnter filename to save (e.g., output.jpg): ").strip()
    if not filename:
        print("[Error] Filename cannot be empty.")
        return
        
    # Ensure images directory exists or save to current
    success = cv2.imwrite(filename, image)
    if success:
        print(f"Successfully saved image as '{filename}'!")
    else:
        print(f"[Error] Failed to save image to '{filename}'. Check the file path.")

def main():
    # Prompt for image file path
    image_path = input("Enter path to your image (e.g., images/example.jpg): ").strip()
    original_image = load_image(image_path)
    
    if original_image is None:
        return

    current_image = original_image.copy()
    processed_result = None

    while True:
        print("\n==========================================")
        print(" Interactive Edge Detection & Filtering")
        print("==========================================")
        print("1. Sobel Edge Detection")
        print("2. Canny Edge Detection")
        print("3. Laplacian Edge Detection")
        print("4. Gaussian Smoothing Filter")
        print("5. Median Filtering")
        print("6. Reset Image to Original")
        print("7. Save Current Processed Image")
        print("8. Exit")
        
        choice = input("Select an option (1-8): ").strip()

        if choice == "1":
            try:
                ksize = int(input("Enter kernel size for Sobel (odd integer, e.g. 3, 5): "))
                processed_result = apply_edge_detection(current_image, method="sobel", ksize=ksize)
                display_side_by_side(original_image, processed_result, f"Sobel Edge Detection (ksize={ksize})")
            except ValueError:
                print("[Error] Please enter a valid integer.")

        elif choice == "2":
            try:
                t1 = int(input("Enter lower threshold for Canny (e.g. 100): "))
                t2 = int(input("Enter upper threshold for Canny (e.g. 200): "))
                processed_result = apply_edge_detection(current_image, method="canny", threshold1=t1, threshold2=t2)
                display_side_by_side(original_image, processed_result, f"Canny Edge Detection ({t1}, {t2})")
            except ValueError:
                print("[Error] Please enter valid integers.")

        elif choice == "3":
            processed_result = apply_edge_detection(current_image, method="laplacian")
            display_side_by_side(original_image, processed_result, "Laplacian Edge Detection")

        elif choice == "4":
            try:
                ksize = int(input("Enter kernel size for Gaussian Blur (odd integer, e.g. 5): "))
                processed_result = apply_filter(current_image, filter_type="gaussian", ksize=ksize)
                display_side_by_side(original_image, processed_result, f"Gaussian Filter (ksize={ksize})")
                current_image = processed_result  # Keep filter chainable
            except ValueError:
                print("[Error] Please enter a valid integer.")

        elif choice == "5":
            try:
                ksize = int(input("Enter kernel size for Median Filter (odd integer, e.g. 5): "))
                processed_result = apply_filter(current_image, filter_type="median", ksize=ksize)
                display_side_by_side(original_image, processed_result, f"Median Filter (ksize={ksize})")
                current_image = processed_result  # Keep filter chainable
            except ValueError:
                print("[Error] Please enter a valid integer.")

        elif choice == "6":
            current_image = original_image.copy()
            processed_result = None
            print("\nImage reset to original state!")

        elif choice == "7":
            save_image(processed_result)

        elif choice == "8":
            print("\nExiting application. Goodbye!")
            break
        else:
            print("[Error] Invalid option. Please choose a number between 1 and 8.")

if __name__ == "__main__":
    main()