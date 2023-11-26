import os
import xml.etree.ElementTree as ET
import cv2
import random
import argparse
import table_detection
import pandas as pd 
#without pre-processing
# from TSR.table_structure_recognition_lines_wol import recognize_structure
# #with pre-processing
# #from table_structure_recognition_all import recognize_structure
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg

import json

def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle

    #print(xA, yA, xB, yB)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def check_bbox(bbox_pred, bbox_true, threshold):
    #print(bbox_pred, bbox_true)

    iou = calc_iou(bbox_pred, bbox_true)
    #print(iou, threshold)
    if(iou > threshold):
        return 1
    return 0

if __name__ == "__main__":


    # cfg = get_cfg() #args.config


    # yaml = "/home/gayathri/g3/table-extraction-mar6/Code-TableDetection/All_X152.yaml"
    # weights = "/home/gayathri/g3/table-extraction-mar6/Code-TableDetection/model_final.pth"
    # #set yaml
    # cfg.merge_from_file(yaml)
    # cfg.MODEL.WEIGHTS = weights
    # predictor = DefaultPredictor(cfg) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset folder")

    args = parser.parse_args()
    folder = args.dataset
    files = os.listdir(folder)
    images = [file for file in files if file.endswith('.png')]
    # xmlfiles = [file for file in files if file.endswith('.xml')]
    table_cells_count=pd.DataFrame(columns=['File','Table','Cells present','Cells predicted',"Threshold"])


    path_root = "/home/ntlpt-52/work/IDP/Table_Extraction/Other_Models/Multi-Type-TD-TSR/Unique_Test_Data/Old_Model/"

    with open(os.path.join(path_root, "ground_truth.json"),'r') as f:
        gtt = json.load(f)
        print("TOTAL FILES IN GTT-> ",len(gtt.keys()))

    with open(os.path.join(path_root, "pred.json"), 'r') as f:
        pred = json.load(f)
        print("TOTAL FILES IN PRED-> ",len(pred.keys()))
 
    
    table_weighted_avg = 0
    cells_weighted_avg = 0
    for thresh in range(5,10):
        threshold = (thresh / 10)
        total_tables = 0
        table_checks = 0
        total_cells=0
        cells_checks=0

        all_files = [file.replace('.png','') for file in images]
        for i in range(len(all_files)):
            # print(all_files[i])

            if f"{all_files[i]}.png" in images:

                try:
                    img_index = images.index(f"{all_files[i]}.png")
                except Exception as e:
                    print(e)
                    pass
                #exit()
                #print(i/2)
                if all_files[i] in gtt.keys() and all_files[i] in pred.keys():
                    img = cv2.imread(os.path.join(folder, images[img_index]))
                    total_tables += len(gtt[all_files[i]])
                    for enum, pred_table in enumerate(pred[all_files[i]]):
                        # pred_bbox = tuple(pred[all_files[i]][enum]['Table'][0].values())
                        bbox_pred = tuple(pred_table['Table'][0].values())
                        # cv2.rectangle(img, (bbox_pred[0], bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), (0, 244, 0), 4)
                        file_pass_flag=False
                        for j, table in enumerate(gtt[all_files[i]]):
                            bbox_true = (table['Table'][0]['x0'], table['Table'][0]['y0'], table['Table'][0]['x1'], table['Table'][0]['y1'])
                            pred_bbox_check=check_bbox(bbox_pred, bbox_true, threshold)
                            if pred_bbox_check and not file_pass_flag:
                                table_checks += pred_bbox_check
                                file_pass_flag=True
                            if pred_bbox_check:
                                try:
                                    total_cells+=len(table['Cell'])
                                    right_cells=0
                                    for pred_cell in pred_table['Cell']:
                                        cell_bbox_pred=tuple(pred_cell.values())
                                        cv2.rectangle(img, (cell_bbox_pred[0], cell_bbox_pred[1]), (cell_bbox_pred[2], cell_bbox_pred[3]), (160, 160, 160), 2)
                                        for ground_cell in table['Cell']:
                                            cell_ground = tuple(ground_cell.values())
                                            cell_bbox_check=check_bbox(cell_bbox_pred,cell_ground,threshold)
                                            right_cells+=cell_bbox_check
                                    table_info=pd.DataFrame({'File':[all_files[i]],'Table':[enum],'Cells present':[len(table['Cell'])],'Cells predicted':[right_cells],'Threshold':threshold})
                                    table_cells_count=pd.concat([table_cells_count,table_info],ignore_index = True)  
                                except:
                                    pass

                                     
                        cells_checks+=right_cells



                    # for enum, table in enumerate(gtt[all_files[i]]):
                    #     print("Image File Name - ",gtt[all_files[i]])
                    #     print("Table - ",enum)
                    #     bbox_true = (table['Table'][0]['x0'], table['Table'][0]['y0'], table['Table'][0]['x1'], table['Table'][0]['y1'])
                    #     try:
                    #         bbox_pred = tuple(pred[all_files[i]][enum]['Table'][0].values())
                    #         print("Pred File - ")
                    #     except:
                    #         bbox_pred = (0, 0, 0, 0)
                    #     cv2.rectangle(img, (bbox_pred[0], bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), (0, 244, 0), 2)
                    #     total += 1
                    #     pred_bbox_check=check_bbox(bbox_pred, bbox_true, threshold)
                    #     checks += pred_bbox_check
                    #     # if pred_bbox_check:
                            
                # cv2.imwrite(os.path.join(path_root, 'PredBbox', images[img_index]), img)


                #print(all_files[i])

                #print(img)

                # try:
                #     table_list, table_coords = table_detection.make_prediction(img, predictor)
                #     #print(table_coords)
                # except Exception as e:
                #     print(e)
                #     continue
                # for j in range(len(table_coords)):
                #     bbox_pred = tuple(table_coords[j])
                #     try : 
                #         coords = gtt[all_files[i]][j] 
                #     except :
                #         pass
                #     bbox_true = (coords['x0'], coords['y0'], coords['x1'], coords['y1'])
                #     total += 1
                #     checks += check_bbox(bbox_pred, bbox_true, threshold)

                
        table_f1 = 0 if(table_checks == 0.0)  else table_checks / total_tables
        print("Table checks {}, Total Tables {}, F1 {}, threshold {}".format(table_checks, total_tables, table_f1, threshold))
        table_weighted_avg += (table_f1 * threshold)/3
        cells_f1 =0 if(cells_checks==0) else cells_checks/total_cells
        print("Cells checks {}, Total Cells {}, F1 {}, threshold {}".format(cells_checks, total_cells, cells_f1, threshold))
        cells_weighted_avg += (cells_f1 * threshold)/3


    print("table weighted_average ", table_weighted_avg)
    print("Cells weighted Avergae ",cells_weighted_avg)
    table_cells_count.to_csv("Table_and_cells_predicted_count.csv")
