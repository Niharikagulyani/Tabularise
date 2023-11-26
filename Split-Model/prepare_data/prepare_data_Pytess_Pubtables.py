"""This module creates crops of tables from the document images,
essentially used for data preparation"""

import os
import glob
import string
import pickle
import argparse
from xml.etree import ElementTree

import cv2
import numpy as np
import pytesseract
from PIL import Image
import json

def apply_ocr(path, image):
    """
    ARGUMENTS:
        path: if ocr data already exists for the given path
        image: Image object that ocr needs to be applied on
    RETURNS:
        bboxes: entire ocr data of the Image
    """

    if os.path.exists(path):
        with open(path, "rb") as f:
            print('ocr path exists')
            return json.load(f)
    else:
        _w, _h = image.size
        _r = 2500 / _w
        image = image.resize((2500, int(_r * _h)))

        print("OCR start")
        ocr = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config="--oem 1"
        )
        print("OCR end")

        bboxes = []
        for i in range(len(ocr["conf"])):
            if ocr["level"][i] > 4 and ocr["text"][i].strip() != "":
                bboxes.append(
                    [
                        len(ocr["text"][i]),
                        ocr["text"][i],
                        int(ocr["left"][i] / _r),
                        int(ocr["top"][i] / _r),
                        int(ocr["left"][i] / _r) + int(ocr["width"][i] / _r),
                        int(ocr["top"][i] / _r) + int(ocr["height"][i] / _r),
                    ]
                )

        bboxes = sorted(
            bboxes, key=lambda box: (box[4] - box[2]) * (box[5] - box[3]), reverse=True
        )
        threshold = np.average(
            [
                (box[4] - box[2]) * (box[5] - box[3])
                for box in bboxes[len(bboxes) // 20 : -len(bboxes) // 4]
            ]
        )
        bboxes = [
            box
            for box in bboxes
            if (box[4] - box[2]) * (box[5] - box[3]) < threshold * 30
        ]

        with open(path, "wb") as f:
            json.dump(bboxes, f)

        return bboxes


def process_files(image_dir, xml_dir, ocr_dir, out_dir):
    """
    ARGUMENTS:
        image_dir: directory of the document image file
        xml_dir: directory of the xml file
        ocr_dir: directory of the ocr file
        out_dir: the output directory for saving data
        
    RETURNS:
        returns no data, saves the processed data to the provided output directory.
    """

    files = [
        file.split("/")[-1].rsplit(".", 1)[0]
        for file in glob.glob(os.path.join(xml_dir, "*.xml"))
    ]
    # print('++++++++++++',files)
    files.sort()

    col_merge_counter = 0
    row_merge_counter = 0

    for ii, file in enumerate(files):

        #file =file.replace(".png",'')
        filename = file
        image_file = os.path.join(image_dir, filename + ".jpg")
        xml_file = os.path.join(xml_dir, filename + ".xml")
        ocr_file = os.path.join(ocr_dir, filename + "_words.json")

        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)

        ocr = apply_ocr(ocr_file, Image.fromarray(img))
        ocr_mask = np.zeros_like(img)
        for word in ocr:
            #txt = word[1].translate(str.maketrans("", "", string.punctuation))
            #if len(txt.strip()) > 0:
            #   cv2.rectangle(ocr_mask, (word[2], word[3]), (word[4], word[5]), 255, -1)
            txt = word['text'].translate(str.maketrans('', '', string.punctuation))
            if len(txt.strip()) > 0:
                x0=int(word['bbox'][0])
                y0=int(word['bbox'][1])
                x1=int(word['bbox'][2])
                y1=int(word['bbox'][3])

                # cv2.rectangle(ocr_mask, (word['bbox'][0], word['bbox'][1]), (word['bbox'][2], word['bbox'][3]), 255, -1)
                cv2.rectangle(ocr_mask, (x0,y0), (x1, y1), 255, -1)

        # cv2.imshow("OCR MASK",ocr_mask)
        # cv2.waitKey(0)


        if (
            os.path.exists(image_file)
            and os.path.exists(xml_file)
            and os.path.exists(ocr_file)
        ):
            print("[", ii, "/", len(files), "]", "Processing: ", file)
            tree = ElementTree.parse(xml_file)
            root = tree.getroot()
            for i, obj in enumerate(root.findall(".//Table")):
                table_name = filename               
                columns = []
                rows = []
                rect = [
                    int(float(obj.attrib["x0"])) if "." in obj.attrib["x0"] else int(obj.attrib["x0"]),
                    int(float(obj.attrib["y0"])) if "." in obj.attrib["y0"] else int(obj.attrib["y0"]),
                    int(float(obj.attrib["x1"])) if "." in obj.attrib["x1"] else int(obj.attrib["x1"]),
                    int(float(obj.attrib["y1"])) if "." in obj.attrib["y1"] else int(obj.attrib["y1"]),
                ]

                img_crop = img[rect[1] : rect[3], rect[0] : rect[2]]
                ocr_mask_crop = ocr_mask[rect[1] : rect[3], rect[0] : rect[2]]
                ocr_mask_crop2 = ocr_mask_crop.copy()

                col_spans = []
                row_spans = []
                if len(obj.findall(".//Column")) <=0:
                    continue
                # Rows and columns append condition is changed as exception occurs when neg values are added in the list
                for col in obj.findall(".//Column"):
                    columns.append(int(float(col.attrib["x0"])) - rect[0] if (int(float(col.attrib["x0"])) - rect[0]) >0 else 0 )
                for row in obj.findall(".//Row"):
                     rows.append(int(float(row.attrib["y0"])) - rect[1] if (int(float(row.attrib["y0"])) - rect[1]) >0 else 0)
                # cv2.imshow("OCR MASK",ocr_mask_crop2)
                # cv2.waitKey(0)
                for cell in obj.findall(".//Cell"):
                    if (
                        cell.attrib["endCol"] != cell.attrib["startCol"]
                        or cell.attrib["endRow"] != cell.attrib["startRow"]
                    ):
                        x0, y0, x1, y1 = map(
                            int,
                            [
                                float(cell.attrib["x0"]),
                                float(cell.attrib["y0"]),
                                float(cell.attrib["x1"]),
                                float(cell.attrib["y1"]),
                            ],
                        )
                        # x0 -= rect[0] - 10
                        # y0 -= rect[1] - 10
                        # x1 -= rect[0] + 10
                        # y1 -= rect[1] + 10
                        #correct coordinates
                        x0 -= rect[0]
                        y0 -= rect[1]
                        x1 -= rect[0]
                        y1 -= rect[1]

                        cell_mask = ocr_mask[y0:y1, x0:x1]
                        row_mask = ocr_mask[y0:y1, :]
                        col_mask = ocr_mask[:, x0:x1]

                        indices = np.where(cell_mask != 0)
                        row_indices = np.where(row_mask != 0)
                        col_indices = np.where(col_mask != 0)

                        if len(indices[0]) != 0:
                            x_min = np.amin(indices[1]) + x0
                            x_max = np.amax(indices[1]) + x0
                            y_min = np.amin(indices[0]) + y0
                            y_max = np.amax(indices[0]) + y0

                            if cell.attrib["endCol"] != cell.attrib["startCol"]:
                                col_spans.append(
                                    (
                                        np.amin(col_indices[1]) + x0,
                                        np.amin(indices[0]) + y0,
                                        np.amax(col_indices[1]) + x0,
                                        np.amax(indices[0]) + y0,
                                    )
                                )
                                col_merge_counter += 1

                            if cell.attrib["endRow"] != cell.attrib["startRow"]:
                                row_spans.append(
                                    (
                                        np.amin(indices[1]) + x0,
                                        np.amin(row_indices[0]) + y0,
                                        np.amax(indices[1]) + x0,
                                        np.amax(row_indices[0]) + y0,
                                    )
                                )
                                row_merge_counter += 1

                        cv2.rectangle(ocr_mask_crop2, (x0, y0), (x1, y1), 0, -1)

                # cv2.imshow("OCR MASK2",ocr_mask_crop2)
                # cv2.waitKey(0)

                bboxes_table = []
                for box in ocr:
                    coords = word['bbox']
                    intrsct = [
                        max(coords[0], rect[0]),
                        max(coords[1], rect[1]),
                        min(coords[2], rect[2]),
                        min(coords[3], rect[3]),
                    ]
                    w = intrsct[2] - intrsct[0]
                    h = intrsct[3] - intrsct[1]

                    w2 = coords[2] - coords[0]
                    h2 = coords[3] - coords[1]
                    if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                        box = word['bbox']
                        box[0] -= rect[0]
                        box[1] -= rect[1]
                        box[2] -= rect[0]
                        box[3] -= rect[1]
                        bboxes_table.append(box)
                ocr = [box for box in ocr if box not in bboxes_table]

                img_crop_masked = img_crop.copy()
                img_crop_masked[ocr_mask_crop == 0] = 255

                col_gt_mask = np.zeros_like(img_crop)
                row_gt_mask = np.zeros_like(img_crop)

                non_zero_rows = np.append(
                    np.where(np.count_nonzero(ocr_mask_crop2, axis=1) != 0)[0],
                    [0, img_crop.shape[0]],
                )
                non_zero_cols = np.append(
                    np.where(np.count_nonzero(ocr_mask_crop2, axis=0) != 0)[0],
                    [0, img_crop.shape[1]],
                )
                # When no cols are added for the image then xml creation is skipped for the image
                if len(columns)==0:
                    continue
                for col in columns:
                    if col == 0 or col == img_crop.shape[1]:
                        continue
                    diff = non_zero_cols - col
                    left = min(-diff[diff < 0]) + 1
                    try:
                        right = min(diff[diff > 0])
                    except:  
                        f = open("demofile2.txt", "a")
                        f.write(filename)
                        f.close()                     
                        right = min(-diff[diff < 0])+1   
                    # min(diff[diff > 0])                                       
                    col_gt_mask[:, col - left : col + right] = 255

                for row in rows:
                    if row == 0 or row == img_crop.shape[0]:
                        continue
                    diff = non_zero_rows - row
                    above = min(-diff[diff < 0]) + 1
                    try:
                        below = min(diff[diff > 0])
                    except:
                        f = open("demofile2.txt", "a")
                        f.write(filename)
                        f.close()                     
                        below = min(-diff[diff < 0])+1
                    #below = min(diff[diff > 0])
                    row_gt_mask[row - above : row + below, :] = 255

                # cv2.imshow("ROW GT MASK",row_gt_mask)
                # cv2.waitKey(0)
                # cv2.imshow("COL GT MASK",col_gt_mask)
                # cv2.waitKey(0)
                # grid= ~ (col_gt_mask | row_gt_mask)
                # cv2.imshow("GRID",grid)
                # cv2.waitKey(0)
                # cv2.imwrite(
                #     os.path.join(out_dir, 'split_labels_plot',table_name + "_col.png"), col_gt_mask
                # )
                # cv2.imwrite(
                #     os.path.join(out_dir,'split_labels_plot', table_name + "_row.png"), row_gt_mask
                # )


                cv2.imwrite(
                    os.path.join(out_dir, "table_images", table_name + ".png"), img_crop
                )

                with open(
                    os.path.join(
                        out_dir, "table_split_labels", table_name + "_row.txt"
                    ),
                    "w",
                ) as f:
                    for i in row_gt_mask[:, 0]:
                        f.write(str(i) + "\n")

                with open(
                    os.path.join(
                        out_dir, "table_split_labels", table_name + "_col.txt"
                    ),
                    "w",
                ) as f:
                    for i in col_gt_mask[0, :]:
                        f.write(str(i) + "\n")

                with open(os.path.join(out_dir, "table_ocr", table_name + ".json"), "w") as f:
                    json.dump(bboxes_table, f)



if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "-img",
        "--image_dir",
        type=str,
        help="Directory containing document-level images",
        default=r'E:\\OX4\deep splerge\Split Merge\Split Merge\\images\\',
        # required=True,
    )

    _parser.add_argument(
        "-xml",
        "--xml_dir",
        type=str,
        help="Directory containing document-level xmls",
        default=r'E:\\OX4\deep splerge\Split Merge\Split Merge\\labels\\',
        # required=True,
    )

    _parser.add_argument(
        "-ocr",
        "--ocr_dir",
        type=str,
        help="Directory containing document-level ocr files. (If an OCR file is not found, it will be generated and saved in this directory for future use)",
        default=r'E:\\OX4\deep splerge\Split Merge\Split Merge\\ocr\\',
        # required=True,
    )

    _parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path of output directory for generated data",
        default=r'E:\\OX4\deep splerge\Split Merge\Split Merge\\prepareout\\',
        # required=True,
    )

    args = _parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "split_labels_plot"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "table_images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "table_split_labels"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "table_ocr"), exist_ok=True)

    process_files(args.image_dir, args.xml_dir, args.ocr_dir, args.out_dir)
