import os 
from xml.etree import ElementTree
import pandas as pd


original_labels_dir = '/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/labels'
prepared_labels_dir = '/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/out/table_split_labels'
 
# with open('Files_to_test_with_GV.txt') as file:
#     xml_files = [line.strip() for line in file]

xml_files = os.listdir(original_labels_dir)

labels_count = pd.DataFrame(columns=['Actual_Rows','Actual_Columns','Rows_gen','Columns_gen'],index=xml_files)

files_with_correct_prepared_labels = []

for xml_file in xml_files:
    try:
        file_name = xml_file.split(".")[0]
        tree = ElementTree.parse(os.path.join(original_labels_dir,xml_file))
        root = tree.getroot()
        for i, obj in enumerate(root.findall(".//Table")):
            rows = obj.findall(".//Row")
            columns  = obj.findall(".//Column")
            labels_count['Actual_Rows'].loc[xml_file]= len(rows)+1
            labels_count['Actual_Columns'].loc[xml_file]= len(columns)+1
    
        row_label_file = os.path.join(prepared_labels_dir,file_name+"_0_row.txt")
        col_label_file = os.path.join(prepared_labels_dir,file_name+"_0_col.txt")

        with open(row_label_file) as file:
            row_txt= [line.strip() for line in file]
        rows_gen=1
        prev = '0' 
        for i,row_value in enumerate(row_txt):
            if i==0 and row_value!='0':
                print("ROWWW  IIIIIIIIIIIIIIIIIIIDIHXI",row_label_file)
            if prev =='255' and row_value=='0':
                rows_gen+=1
            prev=row_value
        labels_count['Rows_gen'].loc[xml_file]= rows_gen

        with open(col_label_file) as file:
            col_txt= [line.strip() for line in file]
        cols_gen=1
        prev='0'
        for i,col_value in enumerate(col_txt):
            if i==0 and col_value!='0':
                print("COLLL IIIIIIIIIIIIIIIIIIIDIHXI",col_label_file)
            if prev =='255' and col_value=='0':
                cols_gen+=1
            prev=col_value
        labels_count['Columns_gen'].loc[xml_file]= cols_gen

        if len(rows)+1 == rows_gen and len(columns)+1 == cols_gen:
            files_with_correct_prepared_labels.append(xml_file)
 
    except:
        pass

print(labels_count)

labels_count.to_csv("LABELS_COUNT.csv")
print("FILES WITH CORRECT LABELS = ",len(files_with_correct_prepared_labels))

with open(str(len(files_with_correct_prepared_labels))+'_CORRECT_LABELS.txt','w') as file:
    for f in files_with_correct_prepared_labels:
        file.write(f)
        file.write('\n')