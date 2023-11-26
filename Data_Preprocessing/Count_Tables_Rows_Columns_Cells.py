import os 
from xml.etree import ElementTree

# xml_directories=['/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table-training/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table training II/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - training iii/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table-test/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - test iii/T-Truth/labels']

xml_directories =['/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/SplitModelData/Test_Data/Labels']
total_tables=0
total_rows=0
total_columns=0
total_cells=0
for xml_dir in xml_directories:
    for xml_file in os.listdir(xml_dir):
        tree = ElementTree.parse(os.path.join(xml_dir,xml_file))
        root = tree.getroot()
        total_tables+=len(root.findall(".//Table"))
        for i, obj in enumerate(root.findall(".//Table")):
            
            total_columns+=len(obj.findall(".//Column"))
            total_rows+=len(obj.findall(".//Row"))
            total_cells+=len(obj.findall(".//Cell"))

print("TOTAL TABLES -> ",total_tables)
print("TOTAL COLUMNS -> ",total_columns)
print("TOTAL ROWS -> ",total_rows)
print("TOTAL CELLS -> ",total_cells)


