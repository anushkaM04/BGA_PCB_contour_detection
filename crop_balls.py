import cv2
import os

def crop_yolo_annotations(image_path, txt_path, output_dir):
    """
    Reads an image and its corresponding YOLOv8 .txt annotation file,
    and saves cropped images of every bounding box.
    """
    # 1. Load the image and get its dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    img_height, img_width = img.shape[:2]

    # 2. Check if the annotation file exists
    if not os.path.exists(txt_path):
        print(f"Error: Could not find annotation file at {txt_path}")
        return

    # 3. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Base name for saving files (e.g., 'my_image' from 'my_image.jpg')
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 4. Read the text file and process each line (each bounding box)
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        print("The annotation file is empty. No crops to perform.")
        return

    print(f"Found {len(lines)} annotations. Cropping...")

    for index, line in enumerate(lines):
        # Parse the YOLO format: class_id, center_x, center_y, width, height
        parts = line.strip().split()
        if len(parts) != 5:
            continue # Skip invalid lines
            
        class_id = parts[0]
        center_x = float(parts[1])
        center_y = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # 5. Convert normalized YOLO coordinates to absolute pixel coordinates
        abs_center_x = int(center_x * img_width)
        abs_center_y = int(center_y * img_height)
        abs_width = int(box_width * img_width)
        abs_height = int(box_height * img_height)

        # 6. Calculate the top-left and bottom-right coordinates for the crop
        # We use max() and min() to ensure the crop box doesn't go outside the image boundaries
        x_min = max(0, int(abs_center_x - (abs_width / 2)))
        y_min = max(0, int(abs_center_y - (abs_height / 2)))
        x_max = min(img_width, int(abs_center_x + (abs_width / 2)))
        y_max = min(img_height, int(abs_center_y + (abs_height / 2)))

        # 7. Perform the crop using numpy array slicing
        cropped_img = img[y_min:y_max, x_min:x_max]

        # 8. Save the cropped image
        # Format: original_name_classID_cropIndex.jpg
        out_filename = f"{base_filename}_class{class_id}_crop{index}.jpg"
        out_filepath = os.path.join(output_dir, out_filename)
        
        cv2.imwrite(out_filepath, cropped_img)
        print(f"Saved: {out_filename}")

    print("Cropping complete!")

# --- Execution ---
if __name__ == "__main__":
    # Replace these paths with your actual file locations
    IMAGE_FILE = "/home/oem/Documents/Intel/for_img_35_DOC1_leg1/train/images/35.jpg"
    ANNOTATION_FILE = "/home/oem/Documents/Intel/for_img_35_DOC1_leg1/train/labels/35.txt"
    OUTPUT_FOLDER = "/home/oem/Documents/Intel/cropped_results"
    
    crop_yolo_annotations(IMAGE_FILE, ANNOTATION_FILE, OUTPUT_FOLDER)