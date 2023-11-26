import os 
from xml.etree import ElementTree
import numpy as np


xml_dir='/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/nt_reannotation_pascal/Train_Data/converted_labels'

def get_bbox(obj):
    rect = [
                    int(eval(obj.attrib["x0"])),
                    int(eval(obj.attrib["y0"])),
                    int(eval(obj.attrib["x1"])),
                    int(eval(obj.attrib["y1"]))
                ]
      
    return rect

def get_iou(table1,table2):
    # coordinates of the area of intersection.
    ix1 = np.maximum(table1[0], table2[0])
    iy1 = np.maximum(table1[1], table2[1])
    ix2 = np.minimum(table1[2], table2[2])
    iy2 = np.minimum(table1[3], table2[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = table1[3] - table1[1] + 1
    gt_width = table1[2] - table1[0] + 1
     
    # Prediction dimensions.
    pd_height = table2[3] - table2[1] + 1
    pd_width = table2[2] - table2[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

dicey_images =[]
for i,xml_file in enumerate(os.listdir(xml_dir)):
    tree = ElementTree.parse(os.path.join(xml_dir,xml_file))
    root = tree.getroot()
    flag= False
    tables = root.findall(".//Table")
    tables_ymin=[]
    flag = False
    for i in range(0,len(tables)-1):
        table1=get_bbox(tables[i])
        # print('table 1 -> ',table1)
        for j in range(i+1,len(tables)):
            table2=get_bbox(tables[j])
            # print('table 2 -> ',table2)
            iou = get_iou(table1,table2)
            # print('IOU -> ',iou)
            if iou>0.1:
                flag = True
                break
        if flag:
            break
    if flag:
        print('tables wrong -> ',xml_file)
        dicey_images.append(xml_file)
        continue

    for i, obj in enumerate(tables):
        rows=[]
        columns=[]
        rect = [
                    int(eval(obj.attrib["x0"])),
                    int(eval(obj.attrib["y0"])),
                    int(eval(obj.attrib["x1"])),
                    int(eval(obj.attrib["y1"]))
                ]
        for col in obj.findall(".//Column"):                    
            columns.append(int(eval(col.attrib["x0"]))-rect[0])

        columns+=[0,rect[2]-rect[0]]
        columns.sort()
        
        # print('column -> ',columns)
        columns_diff= np.diff(columns)      
        # print('columns difference -> ',columns_diff)
        
        for row in obj.findall(".//Row"):
            rows.append(int(eval(row.attrib["y0"]))-rect[1])

        rows+=[0,rect[3]-rect[1]]
        rows.sort()
        # print('row -> ',rows)
        rows_diff = np.diff(rows)
        # print('rows difference -> ',rows_diff)

        if any(ele <5 for ele in columns_diff):
            print(xml_file)
            print("COLUMNS -> ",columns)
            flag= True

        if any(ele <5 for ele in rows_diff):
            print(xml_file)
            print('ROWS -> ',rows)
            flag=True
    if flag:
        dicey_images.append(xml_file)

print("total -> ",dicey_images)

# with open('Wrong_Annotated_Files.txt','w') as file:
#     for img in dicey_images:
#         file.write(img)
#         file.write('\n')
        
        
        


        
