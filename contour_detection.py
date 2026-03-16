import cv2
import numpy as np
import os

def analyze_bga_complete(image_path, annotation_path, output_folder):
    # 1. Setup
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    img_h, img_w = img.shape[:2]

    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found at {annotation_path}")
        return

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    # Configuration
    threshold_val = 95
    current_batch_idx = 0
    balls_per_batch = 16  # 4x4 grid
    grid_size = 4
    cell_w, cell_h = 200, 200

    print("--- BGA BATCH TUNER LOADED ---")
    print(f"Total solder balls found: {len(lines)}")
    print("CONTROLS: [N]ext Batch | [P]revious Batch | [ENTER] Thresh+1 | [BACKSPACE] Thresh-1 | [S]ave & Exit")

    # Initialize the window explicitly
    cv2.namedWindow("BGA Batch Tuner", cv2.WINDOW_AUTOSIZE)

    while True:
        # Create a blank black canvas for the grid
        display_grid = np.zeros((cell_h * grid_size + 40, cell_w * grid_size, 3), dtype=np.uint8)

        start_idx = current_batch_idx * balls_per_batch
        end_idx = min(start_idx + balls_per_batch, len(lines))
        batch_lines = lines[start_idx:end_idx]

        for i, line in enumerate(batch_lines):
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            x_c, y_c, w_n, h_n = map(float, parts[1:])
            
            x1, y1 = max(0, int((x_c - w_n/2) * img_w)), max(0, int((y_c - h_n/2) * img_h))
            x2, y2 = min(img_w, int((x_c + w_n/2) * img_w)), min(img_h, int((y_c + h_n/2) * img_h))

            roi = img[y1:y2, x1:x2].copy()
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Void Detection Logic
            _, ball_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, pot_voids = cv2.threshold(roi_gray, threshold_val, 255, cv2.THRESH_BINARY)
            act_voids = cv2.bitwise_and(pot_voids, pot_voids, mask=ball_mask)
            cnts, _ = cv2.findContours(act_voids, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                if cv2.contourArea(cnt) > 2:
                    cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 1)

            r, c = divmod(i, grid_size)
            roi_resized = cv2.resize(roi, (cell_w, cell_h))
            display_grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = roi_resized
            cv2.putText(display_grid, f"ID:{start_idx+i}", (c*cell_w+5, r*cell_h+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # UI Text overlay at the bottom
        status_text = f"Batch: {current_batch_idx + 1} | Thresh: {threshold_val} | Balls: {start_idx}-{end_idx-1}"
        cv2.putText(display_grid, status_text, (10, cell_h * grid_size + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FIXED ORDER: Show window FIRST, then set Title
        cv2.imshow("BGA Batch Tuner", display_grid)
        cv2.setWindowTitle("BGA Batch Tuner", status_text)

        key = cv2.waitKey(0)

        if key == 13: # Enter
            threshold_val += 1
        elif key == 8 or key == 255: # Backspace (255 for some Linux envs)
            threshold_val -= 1
        elif key == ord('n') or key == ord('N'):
            if end_idx < len(lines):
                current_batch_idx += 1
        elif key == ord('p') or key == ord('P'):
            if current_batch_idx > 0:
                current_batch_idx -= 1
        elif key == ord('s') or key == ord('S'):
            print(f"\n[SAVING] Applying Threshold {threshold_val} to all balls...")
            break
        elif key == 27: # ESC
            cv2.destroyAllWindows()
            return

    # --- FINAL PROCESSING & SAVING ---
    final_img = img.copy()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        x_c, y_c, w_n, h_n = map(float, parts[1:])
        x1, y1 = max(0, int((x_c - w_n/2) * img_w)), max(0, int((y_c - h_n/2) * img_h))
        x2, y2 = min(img_w, int((x_c + w_n/2) * img_w)) , min(img_h, int((y_c + h_n/2) * img_h))

        roi = final_img[y1:y2, x1:x2].copy()
        rg = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        _, bm = cv2.threshold(rg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, pv = cv2.threshold(rg, threshold_val, 255, cv2.THRESH_BINARY)
        av = cv2.bitwise_and(pv, pv, mask=bm)
        cnts, _ = cv2.findContours(av, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_void_area = sum(cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 2)
        cv2.drawContours(roi, cnts, -1, (0, 255, 0), 1)
        final_img[y1:y2, x1:x2] = roi
        
        cv2.imwrite(os.path.join(output_folder, f"ball_{i}_vArea_{int(total_void_area)}.jpg"), roi)

    cv2.imwrite(os.path.join(output_folder, "A_FULL_BOARD_RESULT.jpg"), final_img)
    cv2.destroyAllWindows()
    print(f"Successfully saved all to: {output_folder}")

# Paths
input_image = "/home/oem/Documents/Intel/Intel_obj.v2i.yolov8/train/images/01_jpg.rf.80566d19cb13634ce2fffab6ac338d84.jpg"
input_txt = "/home/oem/Documents/Intel/Intel_obj.v2i.yolov8/train/labels/01_jpg.rf.80566d19cb13634ce2fffab6ac338d84.txt"
output_dir = "/home/oem/Documents/Intel"

analyze_bga_complete(input_image, input_txt, output_dir)