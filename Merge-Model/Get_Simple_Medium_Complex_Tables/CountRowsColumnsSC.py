import os 
import xml.etree.ElementTree as ET
import pandas as pd 

# with open('/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/PubTablesData/Merges.txt') as file:
#     image_files = [line.strip().replace(".png","") for line in file]
images_dir ='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Prepared_Data/TableImages'
image_files=[file.replace(".png","") for file in os.listdir(images_dir)]

labels_dir ='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/converted_ground_truth/'

elements_count=pd.DataFrame(columns=['Rows_Count','Columns_Count','SpanningCells_Count'],index=image_files)

for file in image_files:
    tree = ET.parse(os.path.join(labels_dir, file+".xml"))
    root=tree.getroot()
    rows = root.findall(".//Row")
    elements_count['Rows_Count'].loc[file] = len(rows)
    columns = root.findall('.//Column')
    elements_count['Columns_Count'].loc[file] =len(columns)
    cells = root.findall('.//Cell')
    spanning_cell_count=0
    for cell in cells:
        if cell.attrib['startRow']!=cell.attrib['endRow'] or cell.attrib['startCol']!=cell.attrib['endCol']:
            spanning_cell_count+=1
    elements_count['SpanningCells_Count'].loc[file] = spanning_cell_count

elements_count.to_csv('/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Elements.csv')

overall_elements_info=open("/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge_Model/deep-splerge/Span_Test_Data/Elements.txt","w")
row_min = elements_count['Rows_Count'].min()
row_min_indices =  elements_count.index[elements_count['Rows_Count']==row_min].tolist()
columns_for_rowMin= elements_count['Columns_Count'][row_min_indices].values
overall_elements_info.writelines("Minimum Rows = "+str(row_min)+" Columns = "+str(columns_for_rowMin)+"\n")

row_max = elements_count['Rows_Count'].max()
row_max_indices =  elements_count.index[elements_count['Rows_Count']==row_max].tolist()
columns_for_rowMax= elements_count['Columns_Count'][row_max_indices].values
overall_elements_info.writelines("Maximum Rows = "+str(row_max)+" Columns = "+str(columns_for_rowMax)+"\n")

row_median = elements_count['Rows_Count'].median()
row_median_indices =  elements_count.index[elements_count['Rows_Count']==row_median].tolist()
columns_for_rowMedian= elements_count['Columns_Count'][row_median_indices].values
overall_elements_info.writelines("Median Rows = "+str(row_median)+" Columns = "+str(columns_for_rowMedian)+"\n")


col_min = elements_count['Columns_Count'].min()
col_min_indices =  elements_count.index[elements_count['Columns_Count']==col_min].tolist()
rows_for_colMin= elements_count['Rows_Count'][col_min_indices].values
overall_elements_info.writelines("Minimum Columns= "+str(col_min)+"Rows = "+str(rows_for_colMin)+"\n")

col_max = elements_count['Columns_Count'].max()
col_max_indices =  elements_count.index[elements_count['Columns_Count']==col_max].tolist()
rows_for_colMax= elements_count['Rows_Count'][col_max_indices].values
overall_elements_info.writelines("Maximum Columns = "+str(col_max)+" Rows = "+str(rows_for_colMax)+"\n")

col_median = elements_count['Columns_Count'].median()
col_median_indices =  elements_count.index[elements_count['Columns_Count']==col_median].tolist()
rows_for_colMedian= elements_count['Rows_Count'][col_median_indices].values
overall_elements_info.writelines("Median Columns = "+str(col_median)+" Rows = "+str(rows_for_colMedian)+"\n")


overall_elements_info.writelines("Minimum Spanning Cells = "+str(elements_count['SpanningCells_Count'].min())+"\n")
overall_elements_info.writelines("Maximum Spanning Cells  = "+str(elements_count['SpanningCells_Count'].max())+"\n")
overall_elements_info.writelines("Median Spanning Cells  = "+str(elements_count['SpanningCells_Count'].median())+"\n")

overall_elements_info.close()
