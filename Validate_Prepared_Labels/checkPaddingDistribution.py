import os
import pandas as pd


def get_padding_size(labels_list):
    padding_sizes=[]
    prev = '0'
    size =0 
    for i,val in enumerate(labels_list):
        if val=='255' and prev == '255':
            size+=1
        elif val=='255' and prev =='0':
            if size !=0:
                padding_sizes.append(size)
            size=1
        prev = val
    padding_sizes.append(size)
    return padding_sizes


labels_dir ='/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/prep_data/table_split_labels'
xml_dir = '/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/labels'

xml_files = [file.replace(",xml","") for file in os.listdir(xml_dir)]
padding_distribution = pd.DataFrame(columns=['Min_Row','Max_Row','Median_Row','Mode_Row','Mean_Row','Min_Col','Max_Col','Median_Col','Mode_Col','Mean_Col'],index=xml_files)

for i,xml_file in enumerate(xml_files):
    try:
        row_label_file = os.path.join(labels_dir,xml_file.replace(".xml","_0_row.txt"))
        col_label_file = os.path.join(labels_dir,xml_file.replace(".xml","_0_col.txt"))
        with open(row_label_file) as file:
                row_list= [line.strip() for line in file]
        with open(col_label_file) as file:
                col_list= [line.strip() for line in file]
        
        padding_height= pd.Series(get_padding_size(row_list))
        padding_width = pd.Series(get_padding_size(col_list))

        padding_distribution['Min_Row'].loc[xml_file]=padding_height.min()
        padding_distribution['Max_Row'].loc[xml_file]=padding_height.max()
        padding_distribution['Median_Row'].loc[xml_file]=padding_height.median()
        padding_distribution['Mode_Row'].loc[xml_file]=padding_height.mode()
        padding_distribution['Mean_Row'].loc[xml_file]=padding_height.mean()
        padding_distribution['Min_Col'].loc[xml_file]=padding_width.min()
        padding_distribution['Max_Col'].loc[xml_file]=padding_width.max()
        padding_distribution['Median_Col'].loc[xml_file]=padding_width.median()
        padding_distribution['Mode_Col'].loc[xml_file]=padding_width.mode()
        padding_distribution['Mean_Col'].loc[xml_file]=padding_width.mean()
    except:
        pass

# padding_distribution.to_csv("Padding_Distribution_GV.csv")

print("mean row padding - ",padding_distribution['Min_Row'].mean())
print("mean col padding - ",padding_distribution['Min_Col'].mean())
    


