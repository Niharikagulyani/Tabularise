import cv2
import os 
import xml.etree.ElementTree as ET
import numpy as np
import shutil

xml_dir ='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Test_Data/Labels'
images_dir='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Test_Data/Images'
output_dir='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Test_Data/Annotations/From_Pubtables'

color_tag= {
    "table":(0,0,204),
    'table column header':(0,0,0),
    'table row':(255,0,0),
    'table column':(255,255,0),
    'table projected row header':(204,0,204),
    'table spanning cell':(170,140,0)
}
images_with_row_header=[]
images_with_spanning_cell=[]

for xml_file  in os.listdir(xml_dir):
    img_file = xml_file.split(".")[0]+".jpg"
    # output_img_dir=os.path.join(output_dir,img_file.split(".")[0])
    # os.makedirs(output_img_dir,exist_ok=True)
    img = cv2.imread(os.path.join(images_dir,img_file))
    masked_img={
    "table":img.copy(),
    'table column header':img.copy(),
    'table row':img.copy(),
    'table column':img.copy(),
    'table projected row header':img.copy(),
    'table spanning cell':img.copy()
    }
    xml_file =img_file.split(".")[0]+".xml"
    tree = ET.parse(os.path.join(xml_dir,xml_file))
    root=tree.getroot()
    for child in root:
        if child.tag=='object':
        #    child_name = child.attrib('name')
        #    bounding_box=child.attrib('bndbox')
        #    print("NAME -> ",child_name)
        #    print("BOUNDING BOX -> ",bounding_box)
           
            name = child.find('name').text


            if name == 'table projected row header' and img_file not in images_with_row_header:
                images_with_row_header.append(img_file)
            elif name == 'table spanning cell' and img_file not in images_with_spanning_cell:
                images_with_spanning_cell.append(img_file)


            bounding_box = child.find('bndbox')
            xmin=int(eval(bounding_box.find('xmin').text))
            ymin=int(eval(bounding_box.find('ymin').text))
            xmax=int(eval(bounding_box.find('xmax').text))
            ymax=int(eval(bounding_box.find('ymax').text))
            print(name)
            print('xmin=',xmin)
            print('ymin=',ymin)
            print('xmax=',xmax)
            print('ymax=',ymax)
            cv2.rectangle(masked_img[name],(xmin,ymin),(xmax,ymax),color_tag[name],4)
            # cv2.imshow(name,masked_img[name])
            # cv2.waitKey(0)
    
    cv2.imwrite(os.path.join(output_dir,img_file.split(".")[0]+"_TABLE_DETECTED.png"),masked_img['table'])
    cv2.imwrite(os.path.join(output_dir,img_file.split(".")[0]+"_COLUMN_HEADER.png"),masked_img['table column header'])
    cv2.imwrite(os.path.join(output_dir,img_file.split(".")[0]+"_ROWS.png"),masked_img['table row'])
    cv2.imwrite(os.path.join(output_dir,img_file.split(".")[0]+"_COLUMNS.png"),masked_img['table column'])
    cv2.imwrite(os.path.join(output_dir,img_file.split(".")[0]+"_PROJECTED_ROW_HEADER.png"),masked_img['table projected row header'])
    cv2.imwrite(os.path.join(output_dir,img_file.split(".")[0]+"_SPANNING_CELL.png"),masked_img['table spanning cell'])



print("IMAGES WITH ROW HEADER -> ",len(images_with_row_header))
print(images_with_row_header)
print("\n\n-----------------------------------------------------------------------\n\n")
print("IMAGES WITH SPANNING CELLS -> ",len(images_with_spanning_cell))
print(images_with_spanning_cell)
comms=[i for i in images_with_spanning_cell if i in images_with_row_header]
print("Images with spanning cells and row header -> ", len(comms))
print(comms)