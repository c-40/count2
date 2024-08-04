# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image
# import requests
# from io import BytesIO

# class ImageSeg:
#     def __init__(self, img, threshold):
#         self.img = img
#         self.threshold = threshold

#     def color_filter(self):
#         # Convert image to HSV color space
#         hsv_img = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2HSV)

#         # Define color range for detecting trees (Adjust values as needed)
#         lower_bound = np.array([30, 40, 40])  # Lower bound for green color in HSV
#         upper_bound = np.array([90, 255, 255])  # Upper bound for green color in HSV

#         # Create a mask for the specified color range
#         mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

#         # Apply the mask to the original image
#         filtered_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

#         return filtered_img

#     def preprocess_img(self, img):
#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
#         _, thresh_img = cv2.threshold(blurred_img, self.threshold, 255, cv2.THRESH_BINARY)
#         return thresh_img

#     def post_process(self, thresh_img):
#         # Apply morphological operations to clean up the binary image
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
#         opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
#         return opened_img

#     def count_trees(self):
#         filtered_img = self.color_filter()
#         thresh_img = self.preprocess_img(filtered_img)
#         processed_img = self.post_process(thresh_img)
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
#         return num_labels - 1  # Subtract 1 to exclude the background

#     def mark_trees(self):
#         filtered_img = self.color_filter()
#         thresh_img = self.preprocess_img(filtered_img)
#         processed_img = self.post_process(thresh_img)
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
#         marked_img = np.array(self.img)

#         for i in range(1, num_labels):  # Skip the background label
#             x, y, w, h, _ = stats[i]
#             cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw red rectangle

#         return marked_img

# def fetch_data_from_api(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Will raise HTTPError for bad responses
#         return response.content
#     except requests.exceptions.HTTPError as err:
#         st.error(f"HTTP error occurred: {err}")
#     except Exception as err:
#         st.error(f"An error occurred: {err}")
#     return None

# def main():
#     st.title("Tree Detection App")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)
#         st.image(img, caption='Uploaded Image', use_column_width=True)

#         threshold = st.slider("Select Threshold", 0, 100, 50)

#         obj = ImageSeg(img, threshold)

#         final_count = 0
#         best_threshold = 0

#         for thresh in range(0, 100, 5):
#             obj.threshold = thresh
#             count = obj.count_trees()
#             if count > final_count:
#                 final_count = count
#                 best_threshold = thresh

#         final_obj = ImageSeg(img, best_threshold)
#         marked_img = final_obj.mark_trees()

#         st.image(marked_img, caption='Marked Trees', use_column_width=True)

#         st.write(f"Final Estimated Tree Count: {final_count} trees")

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

class ImageSeg:
    def __init__(self, img, threshold):
        self.img = img.convert('RGB')  # Ensure the image is in RGB mode
        self.threshold = threshold

    def convert_to_grayscale(self):
        # Convert the image to a numpy array and then to grayscale
        gray_img = np.array(self.img.convert('L'))  # Convert to grayscale
        return gray_img

    def preprocess_img(self, gray_img):
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, thresh_img = cv2.threshold(blurred_img, self.threshold, 255, cv2.THRESH_BINARY)
        return thresh_img

    def post_process(self, thresh_img):
        # Apply morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
        return opened_img

    def count_trees(self):
        gray_img = self.convert_to_grayscale()
        thresh_img = self.preprocess_img(gray_img)
        processed_img = self.post_process(thresh_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        return num_labels - 1  # Subtract 1 to exclude the background

    def mark_trees(self):
        gray_img = self.convert_to_grayscale()
        thresh_img = self.preprocess_img(gray_img)
        processed_img = self.post_process(thresh_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        marked_img = np.array(self.img.convert('RGB'))  # Convert back to RGB for visualization

        for i in range(1, num_labels):  # Skip the background label
            x, y, w, h, _ = stats[i]
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw red rectangle

        return marked_img

def fetch_data_from_api(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise HTTPError for bad responses
        return response.content
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error occurred: {err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")
    return None

def main():
    st.title("Tree Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        threshold = st.slider("Select Threshold", 0, 100, 50)

        obj = ImageSeg(img, threshold)

        final_count = 0
        best_threshold = 0

        for thresh in range(0, 100, 5):
            obj.threshold = thresh
            count = obj.count_trees()
            if count > final_count:
                final_count = count
                best_threshold = thresh

        final_obj = ImageSeg(img, best_threshold)
        marked_img = final_obj.mark_trees()

        st.image(marked_img, caption='Marked Trees', use_column_width=True)

        st.write(f"Final Estimated Tree Count: {final_count} trees")

if __name__ == "__main__":
    main()

