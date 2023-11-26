import os 
import argparse
import cv2
import glob


def recognize_structure(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    # inverting the image
    img_bin_inv = 255 - img_bin
    ##################################

    # countcol(width) of kernel as 100th of total width

    kernel_len_ver = max(10,img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver)) #shape (kernel_len, 1) inverted! xD
    #print("ver", ver_kernel)
    #print(ver_kernel.shape)

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1)) #shape (1,kernel_ken) xD
    #print("hor", hor_kernel)
    #print(hor_kernel.shape)
     # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
     # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.dilate(img_vh, kernel, iterations=5)
    thresh, img_vh = (cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY ))
    bitor = cv2.bitwise_or(img_bin, img_vh)

    #img_median = cv2.medianBlur(bitor, 3)
    img_median = bitor
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, img_height*2)) #shape (kernel_len, 1) inverted! xD
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)
    cv2.imshow('vertical_lines',vertical_lines)
    cv2.waitKey(0)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 3)) #shape (kernel_len, 1) inverted! xD
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    cv2.imshow('horizontal_lines',horizontal_lines)
    cv2.waitKey(0)




def process_files(image_dir):
    """
    ARGUMENTS:
        image_dir: directory of the document image file
        xml_dir: directory of the xml file
        out_dir: the output directory for saving data
        
    RETURNS:
        returns no data, saves the processed data to the provided output directory.
    """

    # with open("/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Files_to_test_with_GV.txt") as file:
    #     files = [line.strip().rsplit(".", 1)[0] for line in file]
    for ii, file in enumerate(os.listdir(image_dir)):
       
        filename = file.split(".")[0]
        image_file = os.path.join(image_dir, file)
        img = cv2.imread(image_file)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        recognize_structure(img)
    return


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "-img",
        "--image_dir",
        type=str,
        help="Directory containing document-level images",
        default=r'/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/images',
        # required=True,
    )

    # _parser.add_argument(
    #     "-xml",
    #     "--xml_dir",
    #     type=str,
    #     help="Directory containing document-level xmls",
    #     default=r'/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/correct_annotation_labels',
    #     # required=True,
    # )

    # _parser.add_argument(
    #     "-o",
    #     "--out_dir",
    #     type=str,
    #     help="Path of output directory for generated data",
    #     default=r'/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/out',
    #     # required=True,
    # )

    args = _parser.parse_args()
    # os.makedirs(args.out_dir, exist_ok=True)
    # os.makedirs(os.path.join(args.out_dir, "Masked_Documents"), exist_ok=True)
    # os.makedirs(os.path.join(args.out_dir, "table_images"), exist_ok=True)
    # os.makedirs(os.path.join(args.out_dir, "table_split_labels"), exist_ok=True)

    # process_files(args.image_dir, args.xml_dir, args.out_dir)
    process_files(args.image_dir)
