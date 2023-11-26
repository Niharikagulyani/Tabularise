import cv2
import os 
import glob
import numpy as np
import argparse

class LINE_DETECTION:

    def line_masking(image):    
        result = image.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        img_height, img_width = gray.shape
        # thresh, img_bin = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        kernel_len_ver = max(10,img_height // 50)
        kernel_len_hor = max(10, img_width // 50)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1)) #shape (1,kernel_ken) xD
        # -----Otsu's threshold------       
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255,255,255), 2) # To remove horizontal lines in original document
    
        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver)) #shape (kernel_len, 1) inverted! xD
        # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255,255,255), 2) # To remove vertical lines in original document
        
        return result

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "-img",
        "--image_dir",
        type=str,
        help="Directory containing document-level images",
        default='/home/ntlpt-52/work/IDP/Layout_Similarity/Logo_Images/images'
    )
    _parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path of output directory for generated data",
        default='/home/ntlpt-52/work/IDP/Layout_Similarity/Logo_Images/Line_Detection_Masked'
    )
    args = _parser.parse_args()
    image_dir = args.image_dir
    out_dir = args.out_dir

    for imagefile in os.listdir(image_dir):
        image_path = os.path.join(image_dir,imagefile)
        image = cv2.imread(image_path)
        # cv2.imshow('img',image)
        # cv2.waitKey(0)
        non_bordered_image = LINE_DETECTION.line_masking(image)
        # cv2.imshow('non_bored_img',non_bordered_image)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(out_dir,imagefile), non_bordered_image)


