import os
import xml.etree.ElementTree as ET
import json
import cv2
from collections import defaultdict


path_root = "/home/ntlpt-52/work/IDP/Table_Extraction/Other_Models/Multi-Type-TD-TSR/Unique_Test_Data"


labels_path = '/home/ntlpt-52/work/IDP/Table_Extraction/Other_Models/Multi-Type-TD-TSR/Unique_Test_Data/New_Model_Output_Reannotation/XML'
img_path = os.path.join(path_root, "Images")

save_image_bbox = "/home/ntlpt-52/work/IDP/Table_Extraction/Other_Models/Multi-Type-TD-TSR/Unique_Test_Data/Pred_BBOX_IMAGES"
os.makedirs(save_image_bbox,exist_ok=True)
files = os.listdir(labels_path)

#file_paths = [os.path.join(labels_path, file) for file in files]

pred = defaultdict(list)

for file in files:
    print(file)
    tree = ET.parse(os.path.join(labels_path, file))
    file = "".join(file[:-6])
    img = cv2.imread(os.path.join(img_path, f"{file}.png"))
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    #print('IMG', img)
    root = tree.getroot()
    for child in root:
        table_info = defaultdict(list) 
        print(child.tag) 
        for table in child:
            if table.tag=='boundingbox':
                x0 = int(float(table.attrib['x']))
                y0 = int(float(table.attrib['y']))
                x1 = x0+int(float(table.attrib['w']))
                y1 = y0+int(float(table.attrib['h']))
                table_info['Table'].append({'x0': x0,
                                            'y0' : y0,
                                            'x1': x1,
                                            'y1' : y1})
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 244, 0), 2)
            elif table.tag=='cell':
                row = table.attrib['row']
                column = table.attrib['column']
                try: 
                    for bb in table:
                        print("__",bb.tag)
                        x0 = int(float(bb.attrib['x']))
                        y0 = int(float(bb.attrib['y']))
                        x1 = x0+int(float(bb.attrib['w']))
                        y1 = y0+int(float(bb.attrib['h']))
                        table_info['Cell'].append({'x0': x0,
                                                    'y0' : y0,
                                                    'x1': x1,
                                                    'y1' : y1,
                                                    'row':row,
                                                    'column':column})
                        # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 4)
                except:
                    pass
        pred[file].append(table_info) 
        """  #print(table.attrib)
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
                        cv2.rectangle(img, (x0, y0), (x1, y1), (204, 0, 204), 4)  
                    elif obj.tag=='Column':
                        table_info['Column'].append({'x0': x0,
                                                    'y0' : y0,
                                                    'x1': x1,
                                                    'y1' : y1})
                        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 51), 4)  
                    elif obj.tag=='Cell':
                        row=int(obj.attrib['startRow'])
                        column= int(obj.attrib['startCol'])
                        table_info['Cell'].append({'x0': x0,
                                                    'y0' : y0,
                                                    'x1': x1,
                                                    'y1' : y1,
                                                    'row':row,
                                                    'column':column})
                        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 4)
                ground_truth[file].append(table_info) """

                        
    #print(img)
    # cv2.imshow('Image',img)
    # cv2.waitKey(0)
    print("SAVING IMAGE")
    cv2.imwrite(os.path.join(save_image_bbox, f"{file}.png"), img)


save_to = os.path.join(path_root, 'pred.json')
with open( save_to,'w') as f:
    json.dump(pred, f, indent=4)

print(f"File saved to {save_to}")