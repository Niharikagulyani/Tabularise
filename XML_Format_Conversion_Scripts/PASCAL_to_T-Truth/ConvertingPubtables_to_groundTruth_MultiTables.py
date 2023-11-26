import cv2
import os 
import xml.etree.ElementTree as ET
import numpy as np
import shutil
from Write_GroundTruth_XML import GroundTruthXML
from collections import defaultdict
import argparse

class Convert:
    def __init__(self) -> None:
        self.Tables=None

    def add_element_to_table(self,element_bbox:list,element_name:str=None) -> None:
        for table in self.Tables:
            table_bbox = self.Tables[table]['bbox']
            diff_x0= element_bbox[0] - table_bbox[0]
            diff_y0= element_bbox[1] - table_bbox[1]
            diff_x1= table_bbox[2] - element_bbox[2]
            diff_y1= table_bbox[3] - element_bbox[3]

            if diff_x0>-3 and diff_y0>-3 and diff_x1>-3 and diff_y1>-3:
                self.Tables[table][element_name].append(element_bbox)
    
    def get_bounding_box(self,element)-> list:
        bounding_box = element.find('bndbox')
        xmin=eval(bounding_box.find('xmin').text)
        ymin=eval(bounding_box.find('ymin').text)
        xmax=eval(bounding_box.find('xmax').text)
        ymax=eval(bounding_box.find('ymax').text)
        
        return [xmin,ymin,xmax,ymax]

    def perform(self,xml_dir,output_dir):

        xml_files=os.listdir(xml_dir)
        for xml_file in xml_files:
            self.Tables=defaultdict(dict)
            print('CONVERTING : ',xml_file)
            tree = ET.parse(os.path.join(xml_dir,xml_file))
            root=tree.getroot()
            fileName=root.find('filename').text
            size = root.find('size')
            width=size.find('width').text
            height=size.find('height').text
            xml_out=GroundTruthXML(filename=fileName,Width=width,Height=height)
            tables= root.findall("./object/[name='table']")
            print('Tables found -> ',len(tables))
        
            for i,table in enumerate(tables):
                table_bbox = self.get_bounding_box(table)
                self.Tables['table_'+str(i)]={
                    'bbox':table_bbox,
                    'rows':[],
                    'columns':[],
                    'span_cells':[]

                }

            rows = root.findall("./object/[name='table row']")
            columns = root.findall("./object/[name='table column']")
            spanning_cells = root.findall("./object/[name='table spanning cell']")

            for row in rows:
                row_bbox = self.get_bounding_box(row)
                self.add_element_to_table(row_bbox,'rows')

            for col in columns:
                col_bbox = self.get_bounding_box(col)
                self.add_element_to_table(col_bbox,'columns')
            
            for span_cell in spanning_cells:
                span_cell_bbox = self.get_bounding_box(span_cell)
                span_cell_bbox=[int(coord) for coord in span_cell_bbox]
                self.add_element_to_table(span_cell_bbox,'span_cells')

            for table in self.Tables:
                table_bbox = self.Tables[table]['bbox']
                rows_list = []
                columns_list = []

                xml_out.add_table(xmin_val=str(table_bbox[0]),ymin_val=str(table_bbox[1]),xmax_val=str(table_bbox[2]),ymax_val=str(table_bbox[3]))
                for row in self.Tables[table]['rows']:
                    row_bbox = row
                    if row_bbox[3]-table_bbox[3]>-4 :
                        continue

                    xml_out.add_row(xmin_val=str(table_bbox[0]),ymin_val=str(row_bbox[3]),xmax_val=str(table_bbox[2]),ymax_val=str(row_bbox[3]))
                    rows_list.append(row_bbox[3])
                
                rows_list+=[table_bbox[1],table_bbox[3]]
                rows_list.sort()

                for col in self.Tables[table]['columns']:
                    col_bbox = col

                    if col_bbox[2]-table_bbox[2]>-4:
                        continue
                    
                    xml_out.add_column(xmin_val=str(col_bbox[2]),ymin_val=str(table_bbox[1]),xmax_val=str(col_bbox[2]),ymax_val=str(table_bbox[3]))
                    columns_list.append(col_bbox[2])
                
                columns_list+=[table_bbox[0],table_bbox[2]]
                columns_list.sort()

                
                

                dont_care = np.full((len(rows_list)-1,len(columns_list)-1),"False")
                rows_int = [int(r) for r in rows_list]
                cols_int = [int(c) for c in columns_list]

                print('Rows list -> ',rows_int)
                print('Cols list -> ',cols_int)
                print('Span Cells -> ',self.Tables[table]['span_cells'])

                span_cells = {}                

                for span_cell in self.Tables[table]['span_cells']:
                    x0_diff = np.array([abs(col_x-span_cell[0]) for col_x in cols_int])
                    y0_diff = np.array([abs(row_y-span_cell[1]) for row_y in rows_int]) 
                    x1_diff = np.array([abs(col_x-span_cell[2]) for col_x in cols_int]) 
                    y1_diff = np.array([abs(row_y-span_cell[3]) for row_y in rows_int])


                    start_row=np.argmin(y0_diff)
                    end_row = np.argmin(y1_diff)-1
                    start_col= np.argmin(x0_diff)
                    end_col = np.argmin(x1_diff)-1
                    span_cells[str(start_row)+"_"+str(start_col)]={
                        'end_row':end_row,
                        'end_col':end_col
                    }

                print('Span Cells list -> ',span_cells)


                for i in range(0,len(rows_list)-1):
                    start_row=i
                    ymin=rows_list[i]
                    
                    for j in range(0,len(columns_list)-1):
                        start_col=j
                        xmin=columns_list[j]
                        
                       
                        if str(start_row)+"_"+str(start_col) in span_cells.keys():
                            end_row = span_cells[str(start_row)+"_"+str(start_col)]['end_row']
                            end_col = span_cells[str(start_row)+"_"+str(start_col)]['end_col']
                            if start_row!=end_row:
                                dont_care[i+1:end_row+1,j]="True"
                                
                            if start_col!=end_col:
                                dont_care[i,j+1:end_col+1]="True"
                            
                            ymax = rows_list[end_row+1]
                            xmax = columns_list[end_col+1]
                        else:
                            end_row=i
                            end_col=j  
                            ymax=rows_list[i+1] 
                            xmax=columns_list[j+1]             
                        xml_out.add_cell(xmin_val=str(xmin),ymin_val=str(ymin),xmax_val=str(xmax),ymax_val=str(ymax),start_row=str(start_row),end_row=str(end_row),start_col=str(start_col),end_col=str(end_col),dont_care=dont_care[i][j])

            xml_out.save_xml(os.path.join(output_dir,xml_file))

                

        # with open('Files_With_Multiple_Tables.txt','w') as file:
        #     for xml_file in files_with_multiple_tables:
        #         file.write(xml_file)
        #         file.write('\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-xml",
        "--xml_dir",
        help="Path to XML Files",
        default='/home/ntlpt-52/Downloads/PubTable-1M/Kaggle-TSR/Multiple_Table_PASCAL',
    )

    parser.add_argument(
        "-o",
        "--output_path",
        help="path to the output directory",
        default='/home/ntlpt-52/Downloads/PubTable-1M/Kaggle-TSR/Multiple_Table_converted'
    )
    configs = parser.parse_args()

    xml_dir = configs.xml_dir
    output_dir= configs.output_path

    os.makedirs(output_dir,exist_ok=True)
    convert = Convert()
    convert.perform(xml_dir=xml_dir,output_dir=output_dir)