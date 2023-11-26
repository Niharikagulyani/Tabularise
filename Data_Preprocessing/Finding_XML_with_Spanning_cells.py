import os 
import xml.etree.ElementTree as ET
import shutil

xml_dir='//home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/RowColumSplit/Split_Model_Dataset/Test_Data/Labels'
dest_dir = '/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/RowColumSplit/Split_Model_Dataset/Training_Data/Spanning_Cells_XML'
spanning_cells_xml=[]

for xml_file in os.listdir(xml_dir):
    tree = ET.parse(os.path.join(xml_dir,xml_file))
    root=tree.getroot()
    for child in root:
        for i, obj in enumerate(root.findall(".//Cell")):
            if obj.attrib['startRow']!=obj.attrib['endRow'] or obj.attrib['startCol']!=obj.attrib['endCol']:
                if xml_file not in spanning_cells_xml:
                    spanning_cells_xml.append(xml_file)
                    # shutil.copy(os.path.join(xml_dir,xml_file),os.path.join(dest_dir,xml_file))


print('FIles with spanning cells -> ', spanning_cells_xml)
print(len(spanning_cells_xml))
print(len(os.listdir(dest_dir)))
