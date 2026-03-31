import cv2
import numpy as np
import os
import glob

def segment_solder_balls(image):
    # Step 1: Slicing & Adaptive Thresholding [cite: 104, 105]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This will correct the ligh&dark unevenness of the X-ray images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)
    
    # Step 2: Circle Detection [cite: 107]
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=50)
    
    #TUNING: param2 changes from 30 to 20
    #(Lower param2 = detects more circles, if false circles come change to 22-25)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=20, minRadius=5, maxRadius=60)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    return []

def radial_scan_voids(ball_img, radius_max):
    # Step 3: Multi-directional Radial Scanning [cite: 162, 169]
    height, width = ball_img.shape
    center_x, center_y = width // 2, height // 2
    void_mask = np.zeros_like(ball_img)
    
    #NEW: Dynamic Thresholding Logic
    # 1. Calculate only the average intensity inside the circle, (avoiding the corners of the circles)
    mask = np.zeros_like(ball_img)
    cv2.circle(mask, (center_x, center_y), radius_max, 255, -1)
    mean_intensity = cv2.mean(ball_img, mask=mask)[0]

    # 2. Voids are mostly brighter as compared to all of the background pixels
    # According to the paper absolute should be difference > 6, but we will take a bit of margin.
    # TUNING: If voids are not getting detected tweak this absolute difference to be > 5 or 10, if you encounter noise tweak it to 20
    intensity_offset = 12

    # Paper suggests scanning from r_max down to 1 [cite: 169]
    for r in range(radius_max, 0, -1):
        # Angular scan from 0 to 360 degrees [cite: 189]
        for angle in range(0, 360, 2):
            theta = np.deg2rad(angle)
            x = int(center_x + r * np.cos(theta))
            y = int(center_y + r * np.sin(theta))
            
            if 0 <= x < width and 0 <= y < height:
                # Void Condition: Intensity check [cite: 195]
                # In X-rays, voids are usually brighter [cite: 52]
                if ball_img[y, x] > (mean_intensity + intensity_offset): # Placeholder for intensity threshold
                    void_mask[y, x] = 255
                    
    # Step 4: Connectivity & Filtering [cite: 223, 224]
    # Remove noise smaller than 9 pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(void_mask, connectivity=8)
    final_mask = np.zeros_like(void_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 9:
            final_mask[labels == i] = 255
            
    return final_mask

def process_pipeline(img_path, base_output_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Error: Image read nahi ho payi: {img_path}")
        return

    # Image ka naam nikalna (e.g., '35.jpg' -> '35')
    img_filename = os.path.basename(img_path)
    img_name_without_ext = os.path.splitext(img_filename)[0]
    
    # 1. Is specific image ke liye alag output directory banayein
    img_output_dir = os.path.join(base_output_dir, img_name_without_ext)
    os.makedirs(img_output_dir, exist_ok=True)
    
    balls = segment_solder_balls(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Original image ki ek copy banayein mapping ke liye
    final_overlay = img.copy()
    
    print(f"🔄 Processing: {img_filename} - Found {len(balls)} balls.")

    for i, (x, y, r) in enumerate(balls):
        # Boundary constraints
        y1, y2 = max(0, y-r), min(gray.shape[0], y+r)
        x1, x2 = max(0, x-r), min(gray.shape[1], x+r)
        
        # Har ball ko crop karna
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        # Void detection apply karna
        voids = radial_scan_voids(crop, r)
        
        # Void Percentage calculate karna [cite: 231]
        void_area = np.sum(voids == 255)
        ball_area = np.pi * (r**2)
        percentage = (void_area / ball_area) * 100
        
        # 3. Individual results save karein
        cv2.imwrite(os.path.join(img_output_dir, f"ball_{i}_original.jpg"), crop)
        cv2.imwrite(os.path.join(img_output_dir, f"ball_{i}_voids.png"), voids)
        
        # 4. Remapping onto Original Image
        void_y_indices, void_x_indices = np.where(voids == 255)
        orig_y = void_y_indices + y1
        orig_x = void_x_indices + x1
        
        final_overlay[orig_y, orig_x] = [0, 0, 255] # Red voids
        
        # Solder ball ko visualize karne ke liye ek blue circle aur text
        cv2.circle(final_overlay, (x, y), r, (255, 0, 0), 1)
        cv2.putText(final_overlay, f"{percentage:.1f}%", (x - int(r*0.8), y - r - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # 5. Final poori remapped image ko save karein
    final_output_path = os.path.join(img_output_dir, f"{img_name_without_ext}_remapped.jpg")
    cv2.imwrite(final_output_path, final_overlay)
    print(f"✅ Saved results for {img_filename} in '{img_output_dir}' folder.")

def process_folder(input_folder, output_folder):
    """Folder ki saari images ko process karne ka main function"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Valid image formats define karein
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    # Folder mein se saari files nikalein
    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"❌ Koi valid image nahi mili '{input_folder}' mein.")
        return

    print(f"🚀 Total {len(image_files)} images detect hui hain. Processing start ho rahi hai...\n")
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        process_pipeline(img_path, output_folder)
        
    print(f"\n🎉 Batch Processing Complete! All results saved in '{output_folder}' ")

# Run the pipeline
if __name__ == "__main__":
    # Yahan apne images wale FOLDER ka path dalein
    input_directory = "/home/oem/Documents/Intel/cropped_results"
    
    # Yahan apne final results save karne wale folder ka path dalein
    output_directory = "/home/oem/Documents/Intel/results_of_new_pipeline"
    
    process_folder(input_directory, output_directory)