import os
import glob
import pickle
import string
import functools

import cv2
import PIL
import numpy as np
import pytesseract
from xml.etree import ElementTree as ET
from base64 import b64encode
from google.cloud import vision


class GenerateTFRecord:
    def __init__(self, imagespath, ocrpath, xmlpath, outxmlpath):
        self.unlvocrpath = ocrpath
        self.unlvimagespath = imagespath
        self.unlvtablepath = xmlpath
        self.outtablepath = outxmlpath
        self.visualizeimgs = False

        self.num_of_max_vertices = 900
        self.max_length_of_word = 30

        self.num_data_dims = 5
        self.max_height = 1024
        self.max_width = 1024

        self.counter = 0
        self.tmp_unlv_tables = None
        self.xmlfilepaths = glob.glob(os.path.join(self.unlvtablepath, "*.xml"))

    def str_to_int(self, str):
        intsarr = np.array([ord(chr) for chr in str])
        padded_arr = np.zeros(shape=(self.max_length_of_word), dtype=np.int64)
        padded_arr[: len(intsarr)] = intsarr
        return padded_arr

    def convert_to_int(self, arr):
        return [int(val) for val in arr]

    def pad_with_zeros(self, arr, shape):
        dummy = np.zeros(shape, dtype=np.int64)
        dummy[: arr.shape[0], : arr.shape[1]] = arr
        return dummy

    def generate_tf_record(
        self,
        im,
        gt_matrices,
        pred_matrices,
        arr,
        tablecategory,
        imgindex,
        output_file_name,
    ):
        """This function generates tfrecord files using given information"""
        gt_matrices = [
            self.pad_with_zeros(
                matrix, (self.num_of_max_vertices, self.num_of_max_vertices)
            ).astype(np.int64)
            for matrix in gt_matrices
        ]
        pred_matrices = [
            self.pad_with_zeros(
                matrix, (self.num_of_max_vertices, self.num_of_max_vertices)
            ).astype(np.int64)
            for matrix in pred_matrices
        ]

        im = im.astype(np.int64)
        img_height, img_width = im.shape
        words_arr = [word['word'] for word in arr]
        no_of_words = len(words_arr)

        lengths_arr = [len(word) for word in words_arr]
        vertex_features = np.zeros(
            shape=(self.num_of_max_vertices, self.num_data_dims), dtype=np.int64
        )
        lengths_arr = np.array(lengths_arr).reshape(len(lengths_arr), -1)
        coordinates_list=[[str(box['x1']),str(box['y1']),str(box['x2']),str(box['y2'])] for box in arr]
        sample_out = np.array(np.concatenate((coordinates_list, lengths_arr), axis=1))
        vertex_features[:no_of_words, :] = sample_out

        vertex_text = np.zeros(
            (self.num_of_max_vertices, self.max_length_of_word), dtype=np.int64
        )
        vertex_text[:no_of_words] = np.array(list(map(self.str_to_int, words_arr)))

        result = {
            "image": im.astype(np.float32),
            "sampled_ground_truths": gt_matrices,
            "sampled_predictions": pred_matrices,
            "sampled_indices": None,
            "global_features": np.array(
                [img_height, img_width, no_of_words, tablecategory]
            ).astype(np.float32),
            "vertex_features": vertex_features.astype(np.float32),
        }

        return result

    @staticmethod
    def apply_ocr(path, image):
        if os.path.exists(path):
            print('ocr path exists - ',path)
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/ntlpt-52/work/IDP/Google_vision_keys/oval-heuristic-387906-1815ed6ee296.json"   
            with open(image, 'rb') as f:
                ctxt = b64encode(f.read()).decode()
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content = ctxt)
            response = client.text_detection(image=image)
            word_coordinates = []
            all_text = ""    
            for i,text in enumerate(response.text_annotations):
                if i != 0:         
                    x1 = min([v.x for v in text.bounding_poly.vertices])
                    x2 = max([v.x for v in text.bounding_poly.vertices])
                    y1 = min([v.y for v in text.bounding_poly.vertices])
                    y2 = max([v.y for v in text.bounding_poly.vertices])   
                    if x2 - x1 == 0:                
                        x2 += 1            
                    if y2 - y1 == 0:
                        y2 += 1            
                    word_coordinates.append({
                        "word": text.description,
                        "left": x1,
                        "top": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2                })
                else:
                    all_text = text.description  

            with open(path, "wb") as f:
                pickle.dump(word_coordinates, f)  
                    
            return word_coordinates
        

    def create_same_matrix(self, arr, ids):
        """Given a list of lists with each list consisting of all ids considered same, this function
         generates a matrix """
        matrix = np.zeros(shape=(ids, ids))
        for subarr in arr:
            for element in subarr:
                matrix[element, subarr] = 1
        return matrix

    def data_generator(self):
        def compare(w1, w2):
            if max(w1['y1'], w2['y1']) - min(w1['y2'], w2['y2']) > 0.2 * (w1['y2'] - w1['y1']):
                if w1['y1'] < w2['y1']:
                    return -1
                elif w1['y1'] > w2['y1']:
                    return 1
                else:
                    return 0
            else:
                if w1['x1'] < w2['x1']:
                    return -1
                elif w1['x1'] > w2['x1']:
                    return 1
                else:
                    return 0

        for counter, filename in enumerate(self.xmlfilepaths):
            print("[", counter, "/", len(self.xmlfilepaths), "] Processing:", filename)
            filename = ".".join(filename.split("/")[-1].split(".")[:-1])
            if not os.path.exists(os.path.join(self.unlvtablepath, filename + ".xml")):
                print("WARNING: Ground truth not found for image ", filename)
                continue
            tree = ET.parse(os.path.join(self.unlvtablepath, filename + ".xml"))
            root = tree.getroot()
            xml_tables = root.findall(".//Table")
            if os.path.exists(os.path.join(self.unlvimagespath, filename + ".png")):
                im = PIL.Image.open(
                    os.path.join(self.unlvimagespath, filename + ".png")
                ).convert("RGB")
            else:
                continue

            bboxes = GenerateTFRecord.apply_ocr(
                os.path.join(self.unlvocrpath, filename + ".pkl"), os.path.join(self.unlvimagespath, filename + ".png")
            )

            for i, obj in enumerate(xml_tables):
                x0 = int(eval(obj.attrib["x0"]))
                y0 = int(eval(obj.attrib["y0"]))
                x1 = int(eval(obj.attrib["x1"]))
                y1 = int(eval(obj.attrib["y1"]))
                im2 = im.crop((x0, y0, x1, y1))

                bboxes_table = []
                for box in bboxes:
                    coords = [box['x1'],box['y1'],box['x2'],box['y2']]
                    intrsct = [
                        max(coords[0], x0),
                        max(coords[1], y0),
                        min(coords[2], x1),
                        min(coords[3], y1),
                    ]
                    w = intrsct[2] - intrsct[0]
                    h = intrsct[3] - intrsct[1]

                    w2 = coords[2] - coords[0]
                    h2 = coords[3] - coords[1]
                    if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                        text = box['word']
                        text = text.translate(
                            str.maketrans("", "", string.punctuation)
                        ).strip()

                        if len(text) == 0:
                            continue

                        if len(box['word']) > self.max_length_of_word:
                            box['word'] = box['word'][: self.max_length_of_word]
                        bboxes_table.append(box)
                bboxes = [box for box in bboxes if box not in bboxes_table]

                bboxes_table.sort(key=functools.cmp_to_key(compare))

                if len(bboxes_table) > self.num_of_max_vertices:
                    print(
                        "\n\nWARNING: Number of vertices (",
                        len(bboxes_table),
                        ")is greater than limit (",
                        self.num_of_max_vertices,
                        ").\n\n",
                    )
                    bboxes_table = bboxes_table[: self.num_of_max_vertices]

                same_cell_boxes = [[] for _ in range(len(obj.findall(".//Cell")))]
                same_row_boxes = [[] for _ in range(len(obj.findall(".//Row")) + 1)]
                same_col_boxes = [[] for _ in range(len(obj.findall(".//Column")) + 1)]
                # print("1. Same cell boxes - ",same_cell_boxes)
                # print("1. Same Row Boxes - ",same_row_boxes)
                # print("1. Same_Col_boxes - ",same_col_boxes)

                try:
                    for idx, cell in enumerate(obj.findall(".//Cell")):
                        if cell.attrib["dontCare"] == "true":
                            continue

                        _x0 = int(eval(cell.attrib["x0"]))
                        _y0 = int(eval(cell.attrib["y0"]))
                        _x1 = int(eval(cell.attrib["x1"]))
                        _y1 = int(eval(cell.attrib["y1"]))
                        for idx2, box in enumerate(bboxes_table):
                            coords =  [box['x1'],box['y1'],box['x2'],box['y2']]

                            intrsct = [
                                max(coords[0], _x0),
                                max(coords[1], _y0),
                                min(coords[2], _x1),
                                min(coords[3], _y1),
                            ]
                            w = intrsct[2] - intrsct[0]
                            h = intrsct[3] - intrsct[1]

                            w2 = coords[2] - coords[0]
                            h2 = coords[3] - coords[1]
                            if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                                same_cell_boxes[idx].append(idx2)

                        # print("idx - ",idx," COL RANGE- ", int(cell.attrib["startCol"])," , ", int(cell.attrib["endCol"]) + 1)

                        for j in range(
                            int(cell.attrib["startCol"]), int(cell.attrib["endCol"]) + 1
                        ):
                            # print(j)
                        
                            same_col_boxes[j] += same_cell_boxes[idx]
                        
                        # print("idx - ",idx," ROW RANGE- ", int(cell.attrib["startRow"])," , " ,int(cell.attrib["endRow"]) + 1)
                        for j in range(
                            int(cell.attrib["startRow"]), int(cell.attrib["endRow"]) + 1
                        ):
                            # print(j)
                            same_row_boxes[j] += same_cell_boxes[idx]
                        
                except Exception as e:
                    print("FILE - ",filename)
                    print(e)
                    
                # print("2. Same cell boxes - ",same_cell_boxes)
                # print("2. Same Row Boxes - ",same_row_boxes)
                # print("2. Same_Col_boxes - ",same_col_boxes)

                gt_matrices = [
                    self.create_same_matrix(same_cell_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_row_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_col_boxes, len(bboxes_table)),
                ]

                table_name = os.path.join(
                    self.outtablepath, filename + "_" + str(i) + ".xml"
                )
                if not os.path.exists(table_name):
                    print('\nERROR: "', table_name, '" not found.')
                    continue
                root_pred = ET.parse(os.path.join(table_name)).getroot()
                table_pred = root_pred.findall(".//Table")[0]

                same_cell_boxes = [
                    [] for _ in range(len(table_pred.findall(".//Cell")))
                ]
                same_row_boxes = [
                    [] for _ in range(len(table_pred.findall(".//Row")) + 1)
                ]
                same_col_boxes = [
                    [] for _ in range(len(table_pred.findall(".//Column")) + 1)
                ]

                for idx, cell in enumerate(table_pred.findall(".//Cell")):
                    if cell.attrib["dontCare"] == "true":
                        continue

                    _x0 = int(eval(cell.attrib["x0"])) + x0
                    _y0 = int(eval(cell.attrib["y0"])) + y0
                    _x1 = int(eval(cell.attrib["x1"])) + x0
                    _y1 = int(eval(cell.attrib["y1"])) + y0
                    for idx2, box in enumerate(bboxes_table):
                        coords =  [box['x1'],box['y1'],box['x2'],box['y2']]

                        intrsct = [
                            max(coords[0], _x0),
                            max(coords[1], _y0),
                            min(coords[2], _x1),
                            min(coords[3], _y1),
                        ]
                        w = intrsct[2] - intrsct[0]
                        h = intrsct[3] - intrsct[1]

                        w2 = coords[2] - coords[0]
                        h2 = coords[3] - coords[1]
                        if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                            same_cell_boxes[idx].append(idx2)

                    for j in range(
                        int(cell.attrib["startCol"]), int(cell.attrib["endCol"]) + 1
                    ):
                        same_col_boxes[j] += same_cell_boxes[idx]
                    for j in range(
                        int(cell.attrib["startRow"]), int(cell.attrib["endRow"]) + 1
                    ):
                        same_row_boxes[j] += same_cell_boxes[idx]

                pred_matrices = [
                    self.create_same_matrix(same_cell_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_row_boxes, len(bboxes_table)),
                    self.create_same_matrix(same_col_boxes, len(bboxes_table)),
                ]

                w_org, h_org = im2.size
                h, w = self.max_height, self.max_width

                if im2.size[0] < 20 or im2.size[1] < 20:
                    continue

                im2 = im2.resize(
                    (im2.size[0] * 2500 // im.size[0], im2.size[1] * 2500 // im.size[0])
                )

                if im2.size[0] > w:
                    im2 = im2.resize((w, im2.size[1] * w // im2.size[0]))
                if im2.size[1] > h:
                    im2 = im2.resize((im2.size[0] * h // im2.size[1], h))

                w_new, h_new = im2.size

                new_im = im2
                # new_im.paste(im2)

                r = w_org / h_org

                for j in range(len(bboxes_table)):
                    bboxes_table[j]['x1'] -= x0
                    bboxes_table[j]['x2'] -= x0
                    bboxes_table[j]['x1'] = bboxes_table[j]['x1'] * w_new // w_org
                    bboxes_table[j]['x2'] = bboxes_table[j]['x2'] * w_new // w_org

                    bboxes_table[j]['y1'] -= y0
                    bboxes_table[j]['y2'] -= y0
                    bboxes_table[j]['y1'] = bboxes_table[j]['y1'] * h_new // h_org
                    bboxes_table[j]['y2'] = bboxes_table[j]['y2'] * h_new // h_org

                if len(bboxes_table) == 0:
                    print(
                        "WARNING: No word boxes found inside table #",
                        i,
                        " in image ",
                        filename,
                    )
                    continue

                img = np.asarray(new_im, np.int64)[:, :, 0]

                gt_matrices = [
                    np.array(matrix, dtype=np.int64) for matrix in gt_matrices
                ]
                pred_matrices = [
                    np.array(matrix, dtype=np.int64) for matrix in pred_matrices
                ]

                yield self.generate_tf_record(
                    img,
                    gt_matrices,
                    pred_matrices,
                    np.array(bboxes_table),
                    0,
                    counter,
                    "_",
                )

