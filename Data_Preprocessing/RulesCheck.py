import os
import cv2
from xml.etree import ElementTree
import pandas as pd

# def unnormalize_box(bbox, width, height): 

#     x_center = float(bbox[0]) * width   
#     y_center = float(bbox[1]) * height   
#     width = float(bbox[2]) * width   
#     height = int(float(bbox[3]) * height)   
#     x0 = int(x_center - (width / 2))   
#     x1 = int(x_center + (width / 2))   
#     y0 = int(y_center - (height / 2))   
#     y1 = int(y_center + (height / 2))   

#     return [x0,y0,x1,y1]

def Rule4(bbox):
    if bbox[0]>=bbox[2]:
        return 'Failed - X coordinates wrong'
    if bbox[1]>=bbox[3]:
        return 'Failed - Y coordinates wrong'
    else:
        return "Passed"
def Rule5(bbox, table_bbox):
    if bbox[0]<table_bbox[0] or bbox[1]<table_bbox[1]:
        return "Failed - (x0,y0) < table's (x0,y0)"
    if bbox[2]>table_bbox[2] or bbox[3]>table_bbox[3]:
        return "Failed - (x1,y1) > table's (x1,y1)"
    if bbox[0]>bbox[2]:
        return "Failed - x0>x1"
    if bbox[1]>bbox[3]:
        return "Failed - y0>y1"
    else:
        return "Passed"


if __name__ == "__main__":
    images_directory='/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/Complete_data/Images/'
    labels_directory='/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/Complete_data/Labels/'
    # os.makedirs('/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/ImagesWithBBox/')
    xml_files = os.listdir(labels_directory)
    rules_df=pd.DataFrame(columns=['Rule-1','Rule-2','Rule-3','Rule-4','Rule-5','Rule-6','Number-Of-Files','Total-Right-Files'],index=xml_files)
    rules_df['Number-Of-Files'].iloc[0]=len(xml_files)
    right_files=0
    for i , xml_file in enumerate(xml_files):
        img = cv2.imread(images_directory+xml_file.split(".")[0]+".png")
        tree = ElementTree.parse(labels_directory+xml_file)
        root = tree.getroot()
        tables= root.findall(".//Table")
        if not tables:
            rules_df['Rule-1'].loc[xml_file]='Failed - Table Not Present'
            continue
        else:
            rules_df['Rule-1'].loc[xml_file]="Passed"
        for i, obj in enumerate(tables):
            flag=False
            table_rect = [
                        int(eval(obj.attrib["x0"])),
                        int(eval(obj.attrib["y0"])),
                        int(eval(obj.attrib["x1"])),
                        int(eval(obj.attrib["y1"])),
                    ]
            
            res = Rule4(table_rect)
            if res!="Passed":
                print("table-rule4-",res)
                rules_df['Rule-4'].loc[xml_file]=str(i+1)+" Table "+res
                continue
            else:
                rules_df['Rule-4'].loc[xml_file]='Table '+res

            cv2.rectangle(img,(table_rect[0],table_rect[1]),(table_rect[2],table_rect[3]),(255,0,0),15)
           
            
            columns= obj.findall(".//Column")
            rows = obj.findall(".//Row")
            cells=obj.findall(".//Cell")
            if not rows and not columns:
                rules_df['Rule-2'].loc[xml_file]="Failed - Rows/Columns Not Present"
                continue
            else:
                rules_df['Rule-2'].loc[xml_file]="Passed"

            if not cells:
                rules_df['Rule-3'].loc[xml_file]="Failed - Cells Not Present"
                continue
            else:
                rules_df['Rule-3'].loc[xml_file]="Passed"

            for i, col in enumerate(columns):
                col_rect=[
                        int(eval(col.attrib["x0"])),
                        int(eval(col.attrib["y0"])),
                        int(eval(col.attrib["x1"])),
                        int(eval(col.attrib["y1"])),
                    ]
                

                res2= Rule5(col_rect,table_rect)
                if res2!="Passed":
                    print(str(i+1)+'column-rule5-',res2)
                    rules_df['Rule-5'].loc[xml_file]=str(i+1)+" Column "+res2
                    flag=True
                    break       
                cv2.rectangle(img,(col_rect[0],col_rect[1]),(col_rect[2],col_rect[3]),(0,51,25),15)
            
            if flag:
                continue

            for i,row in enumerate(rows):
                row_rect=[
                        int(eval(row.attrib["x0"])),
                        int(eval(row.attrib["y0"])),
                        int(eval(row.attrib["x1"])),
                        int(eval(row.attrib["y1"])),
                    ]
                

                res2= Rule5(row_rect,table_rect)
                if res2!="Passed":
                    print(str(i+1)+'row-rule5-',res2)
                    rules_df['Rule-5'].loc[xml_file]=str(i+1)+" Row "+res2
                    flag=True
                    break
                cv2.rectangle(img,(row_rect[0],row_rect[1]),(row_rect[2],row_rect[3]),(0,0,255),15)

            if flag:
                continue
            
           
            
            for i,cell in enumerate(cells):
                cell_rect=[
                        int(eval(cell.attrib["x0"])),
                        int(eval(cell.attrib["y0"])),
                        int(eval(cell.attrib["x1"])),
                        int(eval(cell.attrib["y1"])),
                    ]
                
                res = Rule4(cell_rect)
                if res!="Passed":
                    print(str(i+1)+'cell-rule4-',res)
                    rules_df['Rule-4'].loc[xml_file]=str(i+1)+" Cell "+res
                    flag=True
                    break
                else:
                    rules_df['Rule-4'].loc[xml_file]=res

                res2= Rule5(cell_rect,table_rect)
                if res2!="Passed":
                    print(str(i+1)+'cell-rule5-',res2)
                    rules_df['Rule-5'].loc[xml_file]=str(i+1)+" Cell "+res2
                    flag=True
                    break
                else:
                    rules_df['Rule-5'].loc[xml_file]=res2
                if i<1 or i>len(columns):
                    i=0
                    prev=cell_rect[0]
                    i+=1
                elif cell_rect[0]>prev:
                    rules_df['Rule-6'].loc[xml_file]="Passed"
                else: 
                    rules_df['Rule-6'].loc[xml_file]="Failed"
                    flag=True
                    break
                cv2.rectangle(img,(cell_rect[0],cell_rect[1]),(cell_rect[2],cell_rect[3]),(255,204,229),4)
            
            if flag:
                continue
            rules_df['Rule-4'].loc[xml_file]='Table and Cells Passed'
            rules_df['Rule-5'].loc[xml_file]='Columns and Rows and Cells Passed'
            right_files+=1

        print(xml_file)
        print(img.shape)
        try:
            cv2.imwrite('/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/Complete_data/ImagesWithBBox/'+xml_file.split('.')[0]+'.png',img)
        except:
            print("img not saved")
    rules_df['Total-Right-Files'].iloc[0]=right_files
    rules_df.to_csv('Iteration1-Complete_data-XML_FILES_RULES_CHECK.csv')


                



        
