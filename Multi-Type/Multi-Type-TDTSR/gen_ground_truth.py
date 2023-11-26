import os
import xml.etree.ElementTree as ET
import json
import cv2
from collections import defaultdict


path_root = "/home/ntlpt-52/work/IDP/Table_Extraction/Other_Models/Multi-Type-TD-TSR/Unique_Test_Data"


labels_path = "/home/ntlpt-52/work/IDP/Table_Extraction/Other_Models/Multi-Type-TD-TSR/Unique_Test_Data/New_Annotated_Labels"
img_path = os.path.join(path_root, "Images")

save_image_bbox = "/home/ntlpt-52/work/IDP/Table_Extraction/Other_Models/Multi-Type-TD-TSR/Unique_Test_Data/Ground_BBOX_IMAGES"
os.makedirs(save_image_bbox,exist_ok=True)

files = os.listdir(labels_path)

#file_paths = [os.path.join(labels_path, file) for file in files]

ground_truth = defaultdict(list)

for file in files:
    print(file)
    tree = ET.parse(os.path.join(labels_path, file))
    file = file.replace('.xml','')
    img = cv2.imread(os.path.join(img_path, f"{file}.png"))
    #print('IMG', img)
    root = tree.getroot()
    for child in root:
        if child.tag == "Tables":
            if child[0].tag ==  "GroundTruth": 
                table_info = defaultdict(list)  
                for tables in child[0]:
                    for i,table in enumerate(tables):
                        #print(table.attrib)
                        x0 = int(float(table.attrib['x0']))
                        y0 = int(float(table.attrib['y0']))
                        x1 = int(float(table.attrib['x1']))
                        y1 = int(float(table.attrib['y1']))
                        # ground_truth[file].append({'table':{'x0': x0,
                        #                             'y0' : y0,
                        #                             'x1': x1,
                        #                             'y1' : y1}})
                        table_info['Table'].append({'x0': x0,
                                                    'y0' : y0,
                                                    'x1': x1,
                                                    'y1' : y1})
                        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 244, 0), 2)
                        cv2.imshow("image",img)
                        cv2.waitKey(0)
                        for obj in table:
                            x0 = int(float(obj.attrib['x0']))
                            y0 = int(float(obj.attrib['y0']))
                            x1 = int(float(obj.attrib['x1']))
                            y1 = int(float(obj.attrib['y1']))
                            if obj.tag=='Row':
                                table_info['Row'].append({'x0': x0,
                                                        'y0' : y0,
                                                        'x1': x1,
                                                        'y1' : y1})
                                # cv2.rectangle(img, (x0, y0), (x1, y1), (204, 0, 204), 4)  
                            elif obj.tag=='Column':
                                table_info['Column'].append({'x0': x0,
                                                            'y0' : y0,
                                                            'x1': x1,
                                                            'y1' : y1})
                                # cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 51), 4)  
                            elif obj.tag=='Cell':
                                row=int(obj.attrib['startRow'])
                                column= int(obj.attrib['startCol'])
                                table_info['Cell'].append({'x0': x0,
                                                            'y0' : y0,
                                                            'x1': x1,
                                                            'y1' : y1,
                                                            'row':row,
                                                            'column':column})
                                # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 4)
                        ground_truth[file].append(table_info)
                        
                    cv2.imshow("image",img)
                    cv2.waitKey(0)
                    cv2.imwrite(os.path.join(save_image_bbox, file+"_"+str(i)+".png"), img)


save_to = os.path.join(path_root, 'ground_truth.json')
with open( save_to,'w') as f:
    json.dump(ground_truth, f, indent=4)

print(f"File saved to {save_to}")