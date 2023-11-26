import cv2
import os 
import xml.etree.ElementTree as ET
import numpy as np
import shutil
from Writing_PUBTABLE_annotations import PubTablesFormat
# import lxml.etree
# import lxml.builder  

xml_dir ='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/RowColumSplit/Split_Model_Dataset/Training_Data/Labels_To_Convert2'
output_dir='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/RowColumSplit/Split_Model_Dataset/Training_Data/Labels_Converted2'
os.makedirs(output_dir,exist_ok=True)
pubtables_tags={
    'Table':'table',
    'Row':'table row',
    'Column':'table column'
}

for xml_file in os.listdir(xml_dir):
    tree = ET.parse(os.path.join(xml_dir,xml_file))
    root=tree.getroot()
    image_name=root.attrib['name']
    w=root.attrib['width']
    h=root.attrib['height']
    xml_out=PubTablesFormat(filename=image_name,path=image_name,database=xml_dir.split("/")[-1],width=w,height=h,depth=0)
    for child in root:
        for i, obj in enumerate(root.findall(".//Table")):
            rect = [obj.attrib["x0"],
                    obj.attrib["y0"],
                    obj.attrib["x1"],
                    obj.attrib["y1"]
                ]
            
            xml_out.add_object(name='table',xmin_val=rect[0],ymin_val=rect[1],xmax_val=rect[2],ymax_val=rect[3])
            rows=0
            # cv2.rectangle(masked_img['Table'],(rect[0],rect[1]),(rect[2],rect[3]),color_tag['Table'],4)
            rows={'id':0,'y0':rect[1]}
            columns={'id':0,'x0':rect[0]}
            cells_id=0
            for rcc in obj:
                x0, y0, x1, y1 =[rcc.attrib["x0"],
                                rcc.attrib["y0"],
                                rcc.attrib["x1"],
                                rcc.attrib["y1"]
                                ]
                if rcc.tag=='Cell':
                    if rcc.attrib['startRow']!=rcc.attrib['endRow'] or rcc.attrib['startCol']!=rcc.attrib['endCol']:
                        xml_out.add_object(name='table spanning cell',xmin_val=x0,ymin_val=y0,xmax_val=x1,ymax_val=y1)
                    if cells_id==0:
                        xml_out.add_object(name='table column',xmin_val=columns['x0'],ymin_val=rect[1],xmax_val=rect[2],ymax_val=rect[3])
                        cells_id+=1
                elif rcc.tag=='Row':
                    if rows['id']==0:
                        xml_out.add_object(name='table column header',xmin_val=x0,ymin_val=rows['y0'],xmax_val=x1,ymax_val=y1)
                    xml_out.add_object(name='table row',xmin_val=x0,ymin_val=rows['y0'],xmax_val=x1,ymax_val=y1) 
                    rows['id']+=1
                    rows['y0']=y0
                elif rcc.tag=='Column':
                    if columns['id']==0:
                        xml_out.add_object(name='table row',xmin_val=rect[0],ymin_val=rows['y0'],xmax_val=rect[2],ymax_val=rect[3])
                    xml_out.add_object(name='table column',xmin_val=columns['x0'],ymin_val=y0,xmax_val=x1,ymax_val=y1 )
                    columns['id']+=1
                    columns['x0']=x0
    xml_out.save_xml(os.path.join(output_dir,xml_file))