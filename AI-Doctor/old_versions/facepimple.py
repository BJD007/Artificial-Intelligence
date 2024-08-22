import cv2
import numpy as np

def detect_pimples(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a range for pimple-like colors (reddish)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks for potential pimple regions
    pimple_mask = cv2.add(mask1, mask2)
    
    # Additional mask for skin tones to filter the background
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 150, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Combine masks to focus on skin areas with potential pimples
    combined_mask = cv2.bitwise_and(skin_mask, pimple_mask)

    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply the combined mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

    # Convert the masked image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for better edge detection
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Print the number of contours found before filtering
    print(f"Number of contours found: {len(contours)}")

    # Filter contours to detect pimples based on area and shape
    min_area = 15  # Minimum area threshold for pimples
    max_area = 200  # Maximum area threshold for pimples
    pimple_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Draw detected pimples on the original image
    detected_image = image.copy()
    for cnt in pimple_contours:
        # Calculate the aspect ratio to filter out elongated shapes
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.5 < aspect_ratio < 1.5:  # Accept contours with aspect ratios near 1 (circular)
            cv2.drawContours(detected_image, [cnt], -1, (0, 255, 0), 2)  # Draw contours in green

    return detected_image, len(pimple_contours), pimple_contours, masked_image, blurred, adaptive_thresh

def diagnose(pimple_count, pimple_contours):
    diagnosis = ""
    if pimple_count == 0:
        diagnosis = "No pimples detected. Skin appears healthy."
    elif pimple_count <= 5:
        diagnosis = "Mild acne detected. Consider a gentle skincare routine."
    elif pimple_count <= 15:
        diagnosis = "Moderate acne detected. A consultation with a dermatologist is recommended."
    else:
        diagnosis = "Severe acne detected. Immediate medical attention is advised."

    # Face mapping insights based on the provided link
    for cnt in pimple_contours:
        # Get the centroid of the contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Determine the area of the face based on coordinates
        if cY < 100:  # Forehead area
            diagnosis += " Forehead acne may indicate issues with the digestive system."
        elif 100 <= cY < 200 and 100 <= cX < 300:  # Between eyebrows
            diagnosis += " Acne between the eyebrows may relate to liver health."
        elif 100 <= cY < 200 and cX >= 300:  # Temples
            diagnosis += " Acne on the temples may indicate kidney issues."
        elif 200 <= cY < 300 and 100 <= cX < 300:  # Nose
            diagnosis += " Acne on the nose may suggest heart-related issues."
        elif 200 <= cY < 300 and cX >= 300:  # Cheeks
            diagnosis += " Cheek acne may relate to stomach or respiratory issues."
        elif cY >= 300:  # Chin area
            diagnosis += " Chin acne may indicate hormonal imbalances."

    return diagnosis

def main():
    # Load an image of the face
    image_path = 'face2.jpg'  # Updated with the path to the uploaded image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image. Please check the path.")
        return
    
    # Detect pimples
    detected_image, pimple_count, pimple_contours, masked_image, blurred, adaptive_thresh = detect_pimples(image)
    
    # Print the shape of the masked image
    print(f"Masked image shape: {masked_image.shape}")

    # Provide diagnosis based on pimple count
    diagnosis = diagnose(pimple_count, pimple_contours)
    
    # Display the results
    cv2.imshow('Detected Pimples', detected_image)
    cv2.imshow('Masked Image', masked_image)
    cv2.imshow('Blurred Image', blurred)
    cv2.imshow('Adaptive Threshold', adaptive_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Pimple Count: {pimple_count}")
    print(f"Diagnosis: {diagnosis}")

if __name__ == "__main__":
    main()
