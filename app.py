import streamlit as st
import cv2
from PIL import Image  
import numpy as np

#main function
def crop_document(image):  
    # Convert the image to grayscale  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
      
    # Apply GaussianBlur to reduce noise and improve edge detection  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
      
    # Use Canny edge detection  
    edged = cv2.Canny(blurred, 50, 150)  
      
    # Find contours  
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
      
    # Sort contours by area and find the largest one  
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  
    document_contour = None  
      
    for contour in contours:  
        # Approximate the contour  
        peri = cv2.arcLength(contour, True)  
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  
          
        # If the contour has four points, we assume it's the document  
        if len(approx) == 4:  
            document_contour = approx  
            break  
      
    if document_contour is None:  
        raise ValueError("Document contour not found")  
      
    # Apply perspective transform to get the top-down view of the document  
    pts = document_contour.reshape(4, 2)  
    rect = np.zeros((4, 2), dtype="float32")  
      
    s = pts.sum(axis=1)  
    rect[0] = pts[np.argmin(s)]  
    rect[2] = pts[np.argmax(s)]  
      
    diff = np.diff(pts, axis=1)  
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  
      
    (tl, tr, br, bl) = rect  
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  
    maxWidth = max(int(widthA), int(widthB))  
      
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  
    maxHeight = max(int(heightA), int(heightB))  
      
    dst = np.array([  
        [0, 0],  
        [maxWidth - 1, 0],  
        [maxWidth - 1, maxHeight - 1],  
        [0, maxHeight - 1]], dtype="float32")  
      
    M = cv2.getPerspectiveTransform(rect, dst)  
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  
      
    return warped  

#frontend
def main():  
    st.title("Document Cropper")  
      
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  
      
    if uploaded_file is not None:  
        image = Image.open(uploaded_file)  
        image = np.array(image)  
          
        st.image(image, caption='Uploaded Image', use_column_width=True)  
          
        if st.button("Crop Document"):  
            try:  
                cropped_image = crop_document(image)  
                st.image(cropped_image, caption='Cropped Image', use_column_width=True)  
                  
                # Allow user to download the cropped image  
                result = Image.fromarray(cropped_image)  
                st.download_button("Download Cropped Image", data=result.tobytes(), file_name="cropped_image.png", mime="image/png")  
            except ValueError as e:  
                st.error(f"Error: {e}")  
  
if __name__ == "__main__":  
    main()  
