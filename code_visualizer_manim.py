from manim import *
import numpy as np

class BGAImageLogicFlow(Scene):
    def construct(self):
        # --- TITLE ---
        title = Text("BGA Void Detection Visual Pipeline", font_size=36, color=BLUE)
        title.to_edge(UP, buff=0.2)
        self.play(Write(title))

        # ==========================================================
        # --- Asset Cropping Logic (Using the generated diagram) ---
        # ==========================================================
        assets_file = "/home/oem/Documents/Intel/BGA X-Ray Image/DOE1/Leg1/01.jpg"

        try:
            full_board = ImageMobject(assets_file).scale(0.4)
            raw_img_data = full_board.get_pixel_array()
            h, w = raw_img_data.shape[:2]

            # Pre-crop assets for the sequence
            roi_img = ImageMobject(raw_img_data[int(h*0.35):int(h*0.65), int(w*0.25):int(w*0.38)]).scale_to_fit_height(1.8)
            gray_img = ImageMobject(raw_img_data[int(h*0.35):int(h*0.65), int(w*0.42):int(w*0.55)]).scale_to_fit_height(1.8)
            otsu_img = ImageMobject(raw_img_data[int(h*0.05):int(h*0.35), int(w*0.58):int(w*0.72)]).scale_to_fit_height(1.5)
            manual_img = ImageMobject(raw_img_data[int(h*0.65):int(h*0.95), int(w*0.58):int(w*0.72)]).scale_to_fit_height(1.5)
            bitwise_img = ImageMobject(raw_img_data[int(h*0.35):int(h*0.65), int(w*0.75):int(w*0.88)]).scale_to_fit_height(1.8)
            final_img = ImageMobject(raw_img_data[int(h*0.35):int(h*0.65), int(w*0.89):int(w*1.00)]).scale_to_fit_height(1.8)
        except FileNotFoundError:
            self.add(Text("Please ensure the image path is correct.", color=RED, font_size=20))
            return

        # ==================================================
        # --- STEP 1: INPUTS (Full Board & Bounding Box) ---
        # ==================================================
        full_board.shift(LEFT * 4 + DOWN * 0.5)
        fb_label = Text("Input: Full Board X-Ray", font_size=16).next_to(full_board, UP, buff=0.1)

        detection_box = Rectangle(width=0.4, height=0.4, color=YELLOW, stroke_width=2).move_to(full_board.get_center())

        self.play(FadeIn(full_board), Write(fb_label))
        self.play(Create(detection_box))
        self.wait(1)

        yolo_label = MarkupText("YOLO Annotation:\n(x, y, w, h)", font_size=12, color=YELLOW).next_to(detection_box, UR, buff=0.1)
        self.play(Write(yolo_label))
        self.wait(1)

        # --- STEP 2: ROI EXTRACTION ---
        roi_img.shift(LEFT * 1.5) 
        roi_label = Text("1. Extracted ROI", font_size=16, color=ORANGE).next_to(roi_img, UP, buff=0.1)

        roi_arrow = Arrow(detection_box.get_right(), roi_img.get_left(), color=ORANGE, buff=0.1)

        self.play(Create(roi_arrow), FadeIn(roi_img), Write(roi_label), FadeOut(yolo_label))
        self.wait(1)

        # --- STEP 3: GRAYSCALE CONVERSION ---
        gray_img.shift(RIGHT * 1.0) 
        gray_label = Text("2. cvtColor(BGR2GRAY)", font_size=16, color=GRAY).next_to(gray_img, UP, buff=0.1)
        
        gray_arrow = Arrow(roi_img.get_right(), gray_img.get_left(), color=GRAY, buff=0.1)

        self.play(Create(gray_arrow), FadeIn(gray_img), Write(gray_label))
        self.wait(1)

        # --- STEP 4: DUAL PATH THRESHOLDING ---
        otsu_img.move_to(RIGHT * 3.5 + UP * 1.5)
        manual_img.move_to(RIGHT * 3.5 + DOWN * 1.5)
        
        otsu_label = MarkupText("3a. cv2.THRESH_OTSU\n(Ball Boundary Mask)", font_size=14, color=TEAL).next_to(otsu_img, UP, buff=0.1)
        manual_label = MarkupText("3b. cv2.threshold(threshold_val)\n(Void Candidates)", font_size=14, color=RED).next_to(manual_img, UP, buff=0.1)

        path_up = Arrow(gray_img.get_right(), otsu_img.get_left(), color=TEAL, buff=0.1)
        path_down = Arrow(gray_img.get_right(), manual_img.get_left(), color=RED, buff=0.1)

        self.play(Create(path_up), Create(path_down))
        self.play(FadeIn(otsu_img), Write(otsu_label), FadeIn(manual_img), Write(manual_label))
        self.wait(2)

        # --- STEP 5: MASKING (BITWISE AND) ---
        bitwise_img.shift(RIGHT * 3.5) 
        bitwise_label = MarkupText("4. cv2.bitwise_and\n(Isolate Internal Voids)", font_size=14, color=PURPLE).next_to(bitwise_img, UP, buff=0.1)

        merge_up = Arrow(otsu_img.get_right(), bitwise_img.get_top(), color=PURPLE, buff=0.1)
        merge_down = Arrow(manual_img.get_right(), bitwise_img.get_bottom(), color=PURPLE, buff=0.1)

        self.play(FadeOut(full_board), FadeOut(fb_label), FadeOut(detection_box), FadeOut(roi_arrow))
        
        self.play(Create(merge_up), Create(merge_down))
        self.play(FadeIn(bitwise_img), Write(bitwise_label))
        self.wait(2)

        # --- STEP 6: CONTOURS & FINAL OUTPUT ---
        final_img.next_to(bitwise_img, RIGHT, buff=0.5)
        final_label = MarkupText("5. cv2.drawContours\n(Area > 2px Filter)", font_size=14, color=GREEN).next_to(final_img, UP, buff=0.1)

        final_arrow = Arrow(bitwise_img.get_right(), final_img.get_left(), color=GREEN, buff=0.1)

        self.play(Create(final_arrow))
        self.play(FadeIn(final_img), Write(final_label))
        
        self.play(Indicate(final_img, color=GREEN, scale_factor=1.2))
        self.wait(3)

        self.play(FadeOut(Group(*self.mobjects)))