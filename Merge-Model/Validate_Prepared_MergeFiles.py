import cv2
import os 
import xml.etree.ElementTree as ET
import numpy as np
import shutil
import pickle

xml_dir='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/converted_ground_truth'
merges_dir='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Prepared_Data/Merges'

xml_with_mismatching_merges=[]

for file in os.listdir(merges_dir):
    with (open(os.path.join(merges_dir,file), "rb")) as openfile:
        objects = pickle.load(openfile)
    number_of_row_spans= len(objects['row'])
    number_of_col_spans= len(objects['col'])
    xml_file = file.replace('pkl','xml')
    tree = ET.parse(os.path.join(xml_dir,xml_file))
    root=tree.getroot()
    Cells = root.findall(".//Cell")
    col_merging_cells=0
    row_merging_cells=0
    for cell in Cells:
        if cell.attrib['startRow']!=cell.attrib['endRow']:
            row_merging_cells+=1
        if cell.attrib['startCol']!=cell.attrib['endCol']:
            col_merging_cells+=1
    if number_of_col_spans != col_merging_cells or number_of_row_spans!=row_merging_cells:
        print(file)
        xml_with_mismatching_merges.append(file)
    
print(len(xml_with_mismatching_merges))



