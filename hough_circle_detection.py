# # Hough circle detection on batch of cropped images, images are croppped using yolov8 based annotations .txt file
# import cv2
# import numpy as np
# import glob
# import os

# def empty_callback(x):
#     """An empty callback required for OpenCV trackbars."""
#     pass

# def process_batch(input_dir, output_dir):
#     # 1. Setup output directory
#     os.makedirs(output_dir, exist_ok=True)

#     # 2. Get list of image paths (looking for jpg and png)
#     image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn][g]'))
    
#     if not image_paths:
#         print(f"No images found in {input_dir}.")
#         return

#     # 3. Setup the interactive window and Trackbars
#     window_name = "Hough Circle Tuner"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     # cv2.createTrackbar(Name, Window, Default Value, Max Value, Callback)
#     cv2.createTrackbar('minDist', window_name, 50, 500, empty_callback)
#     cv2.createTrackbar('param1', window_name, 100, 300, empty_callback)
#     cv2.createTrackbar('param2', window_name, 30, 150, empty_callback)
#     cv2.createTrackbar('minRadius', window_name, 10, 200, empty_callback)
#     cv2.createTrackbar('maxRadius', window_name, 100, 500, empty_callback)

#     # 4. Loop through the batch of images
#     for img_path in image_paths:
#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         # Convert to grayscale and blur slightly to reduce false positives
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.medianBlur(gray, 5)

#         print(f"\nCurrently Processing: {os.path.basename(img_path)}")
#         print("Keyboard Controls: [s] Save & Next | [n] Skip to Next | [q] Quit script")

#         # 5. Live preview loop for the current image
#         while True:
#             # Grab current trackbar positions (ensure minimums to avoid math errors)
#             minDist = max(1, cv2.getTrackbarPos('minDist', window_name))
#             param1 = max(1, cv2.getTrackbarPos('param1', window_name))
#             param2 = max(1, cv2.getTrackbarPos('param2', window_name))
#             minR = cv2.getTrackbarPos('minRadius', window_name)
#             maxR = max(minR + 1, cv2.getTrackbarPos('maxRadius', window_name))

#             display_img = img.copy()

#             # Apply Hough Circle Transform with live parameters
#             circles = cv2.HoughCircles(
#                 gray,
#                 cv2.HOUGH_GRADIENT,
#                 dp=1.2, # Resolution inverse ratio (usually 1 or 1.2 is fine)
#                 minDist=minDist,
#                 param1=param1,
#                 param2=param2,
#                 minRadius=minR,
#                 maxRadius=maxR
#             )

#             # Draw circles if any are found
#             if circles is not None:
#                 circles = np.round(circles[0, :]).astype("int")
#                 for (x, y, r) in circles:
#                     # Draw outer edge (Green)
#                     cv2.circle(display_img, (x, y), r, (0, 255, 0), 2)
#                     # Draw center point (Red)
#                     cv2.circle(display_img, (x, y), 2, (0, 0, 255), 3)

#             # Show the live preview
#             cv2.imshow(window_name, display_img)

#             # 6. Keyboard event handling (updates every 1ms)
#             key = cv2.waitKey(1) & 0xFF

#             if key == ord('s'):
#                 out_path = os.path.join(output_dir, "detected_" + os.path.basename(img_path))
#                 cv2.imwrite(out_path, display_img)
#                 print(f"--> Saved to: {out_path}")
#                 break # Break live loop, move to next image
            
#             elif key == ord('n'):
#                 print("--> Skipped image.")
#                 break # Break live loop, move to next image
            
#             elif key == ord('q'):
#                 print("--> Quitting early...")
#                 cv2.destroyAllWindows()
#                 return # Exit the entire function

#     cv2.destroyAllWindows()
#     print("\nBatch processing complete.")

# # --- Execution ---
# if __name__ == "__main__":
#     # Create these folders in your working directory and put your images in the input folder
#     INPUT_FOLDER = "/home/oem/Documents/Intel/cropped_results"
#     OUTPUT_FOLDER = "/home/oem/Documents/Intel/results_hough_on_cropped_img"
    
#     process_batch(INPUT_FOLDER, OUTPUT_FOLDER)



import cv2
import numpy as np
import sys
import os
from scipy.spatial import KDTree  # pip install scipy

# ── INPUT ───────────────────────────────────
IMAGE_PATH = "/home/oem/Documents/Intel/for_img_35_DOC1_leg1/train/images/35.jpg"
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]

base       = os.path.splitext(IMAGE_PATH)[0]
OUTPUT_IMG = base + "_detected.png"
OUTPUT_TXT = base + "_annotations.txt"

# ── HOUGH PARAMETERS (unchanged) ────────────
BLUR_KSIZE       = 5
HOUGH_DP         = 1
HOUGH_MIN_DIST   = 18
HOUGH_PARAM1     = 60
HOUGH_PARAM2     = 22
HOUGH_MIN_RADIUS = 7
HOUGH_MAX_RADIUS = 22

# ── FILTER THRESHOLDS — tune these ──────────
MAX_MEAN_INTENSITY  = 250   # Filter 1: reject circles brighter than this (0-255)
MIN_STD_DEV         = 8.0   # Filter 2: reject circles with flat interior (too uniform)
MIN_CIRCULARITY     = 0.55  # Filter 3: reject non-circular blobs (1.0 = perfect circle)
MIN_NEIGHBORS       = 2     # Filter 4: reject circles with fewer than this many neighbours
NEIGHBOR_RADIUS     = 55    # Filter 4: search radius for neighbours (pixels)
# ────────────────────────────────────────────

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"[ERROR] Could not read: {IMAGE_PATH}")
    sys.exit(1)

h, w   = img.shape[:2]
gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur   = cv2.medianBlur(gray, BLUR_KSIZE)

# ── Step 1: CLAHE — improves contrast before Hough ──
clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

circles = cv2.HoughCircles(
    enhanced,
    cv2.HOUGH_GRADIENT,
    dp        = HOUGH_DP,
    minDist   = HOUGH_MIN_DIST,
    param1    = HOUGH_PARAM1,
    param2    = HOUGH_PARAM2,
    minRadius = HOUGH_MIN_RADIUS,
    maxRadius = HOUGH_MAX_RADIUS,
)

if circles is None:
    print("No circles detected.")
    sys.exit(0)

circles_raw = np.round(circles[0]).astype(int)
print(f"Raw detections     : {len(circles_raw)}")

# ── Helper: extract circular ROI mask ───────
def circle_mask(gray_img, cx, cy, r):
    """Returns pixel values inside a circular mask."""
    mask = np.zeros(gray_img.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return gray_img[mask == 255]

# ── Helper: compute circularity from binary mask ─
def circularity_score(gray_img, cx, cy, r):
    """
    Threshold the circle ROI, find the dominant contour,
    return circularity = 4π·Area / Perimeter².
    1.0 = perfect circle, lower = more irregular.
    """
    roi_mask = np.zeros(gray_img.shape, dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), r, 255, -1)
    roi = cv2.bitwise_and(gray_img, gray_img, mask=roi_mask)

    # Crop to bounding box for speed
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r, gray_img.shape[1]), min(cy + r, gray_img.shape[0])
    crop    = roi[y1:y2, x1:x2]

    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    if peri == 0:
        return 0.0

    return (4 * np.pi * area) / (peri ** 2)

# ── FILTER PASS ─────────────────────────────
kept    = []
reasons = {"intensity": 0, "std_dev": 0, "circularity": 0, "isolated": 0}

for (cx, cy, r) in circles_raw:
    pixels = circle_mask(gray, cx, cy, r)

    # ── Filter 1: Intensity gate ─────────────
    # Real solder balls are dark; reject bright circles
    mean_val = float(np.mean(pixels))
    if mean_val > MAX_MEAN_INTENSITY:
        reasons["intensity"] += 1
        continue

    # ── Filter 2: Texture / std-dev gate ─────
    # Real balls have internal contrast; flat regions don't
    std_val = float(np.std(pixels))
    if std_val < MIN_STD_DEV:
        reasons["std_dev"] += 1
        continue

    # ── Filter 3: Circularity gate ───────────
    # Real balls are round; edge connectors and traces aren't
    circ = circularity_score(gray, cx, cy, r)
    if circ < MIN_CIRCULARITY:
        reasons["circularity"] += 1
        continue

    kept.append((cx, cy, r))

# ── Filter 4: Grid regularity — needs all survivors first ──
# BGA balls form a grid; isolated detections are spurious
if len(kept) >= MIN_NEIGHBORS + 1:
    centers = np.array([(cx, cy) for cx, cy, _ in kept], dtype=float)
    tree    = KDTree(centers)

    final = []
    for i, (cx, cy, r) in enumerate(kept):
        # Count neighbours within NEIGHBOR_RADIUS (excluding self, index 0)
        idxs = tree.query_ball_point([cx, cy], NEIGHBOR_RADIUS)
        n_neighbors = len(idxs) - 1   # subtract self
        if n_neighbors < MIN_NEIGHBORS:
            reasons["isolated"] += 1
            continue
        final.append((cx, cy, r))
else:
    final = kept

# ── Report ───────────────────────────────────
print(f"After intensity     : -{reasons['intensity']}")
print(f"After std-dev       : -{reasons['std_dev']}")
print(f"After circularity   : -{reasons['circularity']}")
print(f"After grid check    : -{reasons['isolated']}")
print(f"Final detections   : {len(final)}")

# ── Draw results ─────────────────────────────
output = img.copy()
for (cx, cy, r) in final:
    cv2.circle(output, (cx, cy), r, (0, 0, 255), 2)
    cv2.circle(output, (cx, cy), 2, (0, 0, 255), -1)

cv2.imwrite(OUTPUT_IMG, output)

# ── Write YOLO annotations ───────────────────
with open(OUTPUT_TXT, "w") as f:
    for (cx, cy, r) in final:
        cx_n = cx / w
        cy_n = cy / h
        w_n  = (2 * r) / w
        h_n  = (2 * r) / h
        f.write(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

print(f"Image saved : {OUTPUT_IMG}")
print(f"Labels saved: {OUTPUT_TXT}")




# import cv2
# import numpy as np
# import sys
# import os

# # ── INPUT ───────────────────────────────────
# IMAGE_PATH = "/home/oem/Documents/Intel/Datset_labels/train/images/test_img_BGA.jpg"   # change this or pass as command-line argument
# # ────────────────────────────────────────────

# # ── TUNING PARAMETERS ───────────────────────
# BLUR_KSIZE       = 5    # median blur kernel (must be odd)
# HOUGH_DP         = 1
# HOUGH_MIN_DIST   = 18   # min px between circle centers
# HOUGH_PARAM1     = 60   # Canny high threshold
# HOUGH_PARAM2     = 22   # accumulator threshold (lower = more circles detected)
# HOUGH_MIN_RADIUS = 7    # min solder ball radius in pixels
# HOUGH_MAX_RADIUS = 22   # max solder ball radius in pixels
# # ────────────────────────────────────────────

# # Accept image path from command line if provided
# if len(sys.argv) > 1:
#     IMAGE_PATH = sys.argv[1]

# # Derive output paths from input filename
# base        = os.path.splitext(IMAGE_PATH)[0]
# OUTPUT_IMG  = base + "_detected.png"
# OUTPUT_TXT  = base + "_annotations.txt"

# # ── Load image ───────────────────────────────
# img = cv2.imread(IMAGE_PATH)
# if img is None:
#     print(f"[ERROR] Could not read image: {IMAGE_PATH}")
#     sys.exit(1)

# h, w = img.shape[:2]
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, BLUR_KSIZE)

# # ── Detect circles ───────────────────────────
# circles = cv2.HoughCircles(
#     blur,
#     cv2.HOUGH_GRADIENT,
#     dp        = HOUGH_DP,
#     minDist   = HOUGH_MIN_DIST,
#     param1    = HOUGH_PARAM1,
#     param2    = HOUGH_PARAM2,
#     minRadius = HOUGH_MIN_RADIUS,
#     maxRadius = HOUGH_MAX_RADIUS,
# )

# if circles is None:
#     print("No circles detected. Try lowering HOUGH_PARAM2.")
#     sys.exit(0)

# circles = np.round(circles[0]).astype(int)

# # ── Draw red circles on output image ─────────
# output = img.copy()
# for (x, y, r) in circles:
#     cv2.circle(output, (x, y), r, (0, 0, 255), 2)    # red outline
#     cv2.circle(output, (x, y), 2, (0, 0, 255), -1)   # red center dot

# cv2.imwrite(OUTPUT_IMG, output)

# # ── Write YOLO-format .txt annotation file ───
# with open(OUTPUT_TXT, "w") as f:
#     for (x, y, r) in circles:
#         cx_n = x / w
#         cy_n = y / h
#         w_n  = (2 * r) / w
#         h_n  = (2 * r) / h
#         f.write(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

# print(f"Detected   : {len(circles)} solder balls")
# print(f"Image out  : {OUTPUT_IMG}")
# print(f"Annot. out : {OUTPUT_TXT}")







# import cv2
# import numpy as np

# img = cv2.imread("/home/oem/Documents/Intel/BGA X-Ray Image/DOE1/Leg1/01.jpg", cv2.IMREAD_GRAYSCALE)        #Total number of solder balls = 174
# img_blur = cv2.medianBlur(img, 5)

# circles = cv2.HoughCircles(
#     img_blur,
#     cv2.HOUGH_GRADIENT,
#     dp=1,
#     minDist=20,          # min distance between ball centers
#     param1=50,
#     param2=30,
#     minRadius=8,         # tune to your ball size in pixels
#     maxRadius=20
# )

# # Convert to YOLO format
# h, w = img.shape
# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")
#     with open("yolo_annotations.txt", "w") as f:
#         for (x, y, r) in circles:
#             # YOLO: class cx cy w h (normalized)
#             cx, cy = x / w, y / h
#             bw, bh = (2 * r) / w, (2 * r) / h
#             f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

# # ### Recommended Strategy
# # ```
# # Phase 1: Run Hough Circles → get ~85-90% of balls auto-detected
# # Phase 2: Upload to Roboflow, use SAM to fix missed/wrong detections  
# # Phase 3: Train YOLOv8 on this clean set → fully automated from here