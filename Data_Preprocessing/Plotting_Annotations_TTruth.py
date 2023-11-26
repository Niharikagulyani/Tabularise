import cv2
import os 
import xml.etree.ElementTree as ET
import numpy as np
import shutil

xml_dir ='/home/ntlpt-52/Downloads/PubTable-1M/OUR_TEST_DATA/Different_Layouts/Labels'
images_dir='/home/ntlpt-52/Downloads/PubTable-1M/OUR_TEST_DATA/Different_Layouts/Images'
output_dir='/home/ntlpt-52/Downloads/PubTable-1M/OUR_TEST_DATA/Different_Layouts/AnnotatedImages'

color_tag= {
    "Table":(0,0,204),
    'Column':(0,0,0),
    'Row':(255,0,0),
    'Cell':(255,255,0)
}
for img_file  in os.listdir(images_dir):
    output_img_dir=os.path.join(output_dir,img_file)
    os.makedirs(output_img_dir,exist_ok=True)
    img = cv2.imread(os.path.join(images_dir,img_file))
    masked_img={
    "Table":img.copy(),
    'Column':img.copy(),
    'Row':img.copy(),
    'Cell':img.copy()
    }
    xml_file =img_file.split(".")[0]+".xml"
    tree = ET.parse(os.path.join(xml_dir,xml_file))
    root=tree.getroot()
    for child in root:
        for i, obj in enumerate(root.findall(".//Table")):
            rect = [
                    int(float(obj.attrib["x0"])),
                    int(float(obj.attrib["y0"])),
                    int(float(obj.attrib["x1"])),
                    int(float(obj.attrib["y1"])),
                ]
            cv2.rectangle(masked_img['Table'],(rect[0],rect[1]),(rect[2],rect[3]),color_tag['Table'],4)
            for rcc in obj:
                x0, y0, x1, y1 = map(
                            int,
                            [
                                float(rcc.attrib["x0"]),
                                float(rcc.attrib["y0"]),
                                float(rcc.attrib["x1"]),
                                float(rcc.attrib["y1"]),
                            ],
                        )
                cv2.rectangle(masked_img[rcc.tag],(x0,y0),(x1,y1),color_tag[rcc.tag],4)            


    
    cv2.imwrite(os.path.join(output_img_dir,"TABLE_DETECTED.jpg"),masked_img['Table'])
    cv2.imwrite(os.path.join(output_img_dir,"CELLS.jpg"),masked_img['Cell'])
    cv2.imwrite(os.path.join(output_img_dir,"ROWS.jpg"),masked_img['Row'])
    cv2.imwrite(os.path.join(output_img_dir,"COLUMNS.jpg"),masked_img['Column'])
