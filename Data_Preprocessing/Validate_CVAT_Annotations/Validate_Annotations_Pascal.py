import os 
from xml.etree import ElementTree
import numpy as np


xml_dir='/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/nt_reannotation_pascal/Krishna/Pascal_VOC_Labels'

def get_bbox(obj):
    bounding_box = obj.find('bndbox')
    xmin=eval(bounding_box.find('xmin').text)
    ymin=eval(bounding_box.find('ymin').text)
    xmax=eval(bounding_box.find('xmax').text)
    ymax=eval(bounding_box.find('ymax').text)
    
    return [xmin,ymin,xmax,ymax]

def get_iou(bbox1,bbox2):
    # coordinates of the area of intersection.
    ix1 = np.maximum(bbox1[0], bbox2[0])
    iy1 = np.maximum(bbox1[1], bbox2[1])
    ix2 = np.minimum(bbox1[2], bbox2[2])
    iy2 = np.minimum(bbox1[3], bbox2[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = bbox1[3] - bbox1[1] + 1
    gt_width = bbox1[2] - bbox1[0] + 1
     
    # Prediction dimensions.
    pd_height = bbox2[3] - bbox2[1] + 1
    pd_width = bbox2[2] - bbox2[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

dicey_images =[]
for i,xml_file in enumerate(os.listdir(xml_dir)):
    print(xml_file)
    tree = ElementTree.parse(os.path.join(xml_dir,xml_file))
    root = tree.getroot()
    flag= False
    tables= root.findall("./object/[name='table']")
    if len(tables)==0:
        dicey_images.append(xml_file)
        continue
    flag = False
    for i in range(0,len(tables)-1):
        table1=get_bbox(tables[i])
        for j in range(i+1,len(tables)):
            table2=get_bbox(tables[j])
            iou = get_iou(table1,table2)
            if iou>0.1:
                flag = True
                break
        if flag:
            break
    if flag:
        print('tables wrong -> ',xml_file)
        dicey_images.append(xml_file)
        continue

    rows = root.findall("./object/[name='table row']")
    columns = root.findall("./object/[name='table column']")

    for row in rows:
        row_bbox = get_bbox(row)
        if row_bbox[3]-row_bbox[1]<5 or row_bbox[2]-row_bbox[0]<5:
            print("ROW BBOX LESS - ",xml_file)
            dicey_images.append(xml_file)
            flag = True
            break
    if flag:
        continue
    for col in columns:
        col_bbox = get_bbox(col)
        if col_bbox[3]-col_bbox[1]<5 or col_bbox[2]-col_bbox[0]<5:
            print("COL BBOX LESS - ",xml_file)
            dicey_images.append(xml_file)
            flag = True
            break
    if flag:
        continue

    if len(tables)>1 :  #not for multiple tables yet:
        continue

    table_bbox = get_bbox(tables[0])
    for row in rows:
        row_bbox = get_bbox(row)
        if row_bbox[3]-table_bbox[3]>3 or table_bbox[1]-row_bbox[1]>3 or row_bbox[2]-table_bbox[2]>3 or table_bbox[0]-row_bbox[0]>3 :
            print('table_bbox - ', table_bbox)
            print("row_bbox - ",row_bbox,' outside the table')
            dicey_images.append(xml_file)
            flag=True
            break

    if flag:
        continue

    for col in columns:
        col_bbox = get_bbox(col)
        if col_bbox[3]-col_bbox[3]>3 or col_bbox[1]-col_bbox[1]>3 or col_bbox[2]-col_bbox[2]>3 or col_bbox[0]-col_bbox[0]>3 :
            print('table_bbox - ', table_bbox)
            print("col_bbox - ",col_bbox,' outside the table')
            dicey_images.append(xml_file)
            flag=True
            break

    if flag:
        continue

print("total -> ",dicey_images)

with open('Krishna_Wrong_Annotated_Files.txt','w') as file:
    for img in dicey_images:
        file.write(img)
        file.write('\n')
        
        
        


        