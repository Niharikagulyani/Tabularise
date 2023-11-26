"""This module creates crops of tables from the document images,
essentially used for data preparation"""

import os
import glob
import string
import pickle
import argparse
from xml.etree import ElementTree
import cv2
import numpy as np
import pandas as pd
from scipy import stats

def get_blank_rows_columns_indices(img,axis,row_col_split,color_scales=None):

    blank_rows_cols_percentage=np.array([]) 
    for i in range(len(row_col_split)-1):
        
        if axis == 1:
            region = img[row_col_split[i]:row_col_split[i+1],:]
            scale = color_scales[i]
        elif axis == 0 : 
            region =img[:,row_col_split[i]:row_col_split[i+1]]
            scale = color_scales[0]

        
        sum_of_pixels = np.sum(region,axis=axis)
        
        if scale ==0:
            i = np.where(sum_of_pixels==0)[0]
            background_color_percentage = np.zeros_like(sum_of_pixels)
            if len(i)>0:
                background_color_percentage[i]=1
        else :
            background_color_percentage = sum_of_pixels/scale
        
        blank_rows_cols_percentage= np.append(blank_rows_cols_percentage,background_color_percentage)

    return blank_rows_cols_percentage


def get_left_or_above(left_list):
    i = len(left_list)-1
    while i>0:
        if left_list[i]-left_list[i-1]==1:
            i=i-1
        else:
            break
    return left_list[i]

def get_right_or_below(right_list):
    i = 0
    while i<len(right_list)-2 :
        if right_list[i+1]-right_list[i]==1:
            i=i+1
        else:
            break
    
    return right_list[i]

def get_neighbour_indices(value,labels_list):
    diff = labels_list - value
    left_index = np.where(diff==-1)[0]
    right_index = np.where(diff==1)[0]
    return left_index,right_index


def get_bg_color(img,indices,axis):
    i=0
    bg_color=[]
    while i<len(indices)-1:
        try:
            if axis == 0:
                region = img[indices[i]:indices[i+1],:]
            elif axis == 1 : 
                region =img[:,indices[i]:indices[i+1]]
            mode = stats.mode(region,axis=None)
            mode = mode.mode[0]
            bg_color.append(mode)
        except:
            pass
        i+=1 
    return bg_color


def get_bg_and_text_color(img,indices,axis,img_bw):
    i=0
    bg_color=[]
    text_color=[]
    # print(img.shape)
    # print(img_bw.shape)
    while i<len(indices)-1:
        
        if axis == 0:
            region = img[indices[i]:indices[i+1],:]
            im_bw= img_bw[indices[i]:indices[i+1],:]
        elif axis == 1 : 
            region =img[:,indices[i]:indices[i+1]]
            im_bw= img_bw[:,indices[i]:indices[i+1]]

        # cv2.imshow('region',region)
        # cv2.waitKey(0)
        # cv2.imshow('region_bw',im_bw)
        # cv2.waitKey(0)

        unique_bw, counts_bw = np.unique(im_bw, return_counts=True)
        # print("FOR BW")
        # print('unique - ',unique_bw)
        # print('counts - ',counts_bw)
        text_bw = unique_bw[int(np.argmin(counts_bw))]
        # print(text_bw)

        # print("for grayscale")
        unique, counts = np.unique(region, return_counts=True)
        # print('Unique - ',unique)
        # print('Counts - ',counts)
        # print('----',len(counts))
        flat = counts.copy()
        flat.sort()
        # print('flat -> ',flat)
        max_occuring = int(flat[-1])
        # second_max_occuring = int(flat[-2])
        max_index = int(np.where(counts==max_occuring)[0][0])
        # print('MAX INDEX - ',max_index)
        # second_max_index = int(np.where(counts == second_max_occuring)[0][0])
        # print("second max index - ",second_max_index)
        bg = unique[max_index]
        # text = unique[second_max_index]
        # print('bg - ',bg)
        # print('text - ',text)
        # cv2.imshow('region',region)
        # cv2.waitKey(0)


        bg_color.append(bg)
        text_color.append(text_bw)
        
        i+=1 
        # exit()
    return bg_color,text_color

def process_files(image_dir, xml_dir, out_dir):
    """
    ARGUMENTS:
        image_dir: directory of the document image file
        xml_dir: directory of the xml file
        out_dir: the output directory for saving data
        
    RETURNS:
        returns no data, saves the processed data to the provided output directory.
    """

    files = [
        file.split("/")[-1].rsplit(".", 1)[0]
        for file in glob.glob(os.path.join(xml_dir, "*.xml"))
    ]

    # with open("/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Files_to_test_with_GV.txt") as file:
    #     files = [line.strip().rsplit(".", 1)[0] for line in file]
    print('++++++++++++',len(files))
    files.sort()

    for ii, file in enumerate(files):
       
        filename = file
        image_file = os.path.join(image_dir, filename + ".png")
        xml_file = os.path.join(xml_dir, filename + ".xml")

        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        if (
            os.path.exists(image_file)
            and os.path.exists(xml_file)
        ):
            print("[", ii, "/", len(files), "]", "Processing: ", file)
            tree = ElementTree.parse(xml_file)
            root = tree.getroot()
            for i, obj in enumerate(root.findall(".//Table")):
                table_name = filename + "_" + str(i)
                rows=[]
                columns=[]
                rect = [
                    int(float(obj.attrib["x0"])) if "." in obj.attrib["x0"] else int(obj.attrib["x0"]),
                    int(float(obj.attrib["y0"])) if "." in obj.attrib["y0"] else int(obj.attrib["y0"]),
                    int(float(obj.attrib["x1"])) if "." in obj.attrib["x1"] else int(obj.attrib["x1"]),
                    int(float(obj.attrib["y1"])) if "." in obj.attrib["y1"] else int(obj.attrib["y1"]),
                ]
              
                img_crop = img[rect[1] : rect[3], rect[0] : rect[2]]
                (thresh, img_crop_bw) = cv2.threshold(img_crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                table_copy = img_crop.copy()
                # print("CROP -> ",img_crop.shape)
                t_height,t_width = img_crop.shape
                cv2.imshow('table',img_crop)
                cv2.waitKey(0)

                if len(obj.findall(".//Column")) <=0:
                    continue
    
                for col in obj.findall(".//Column"):                    
                    columns.append(int(float(col.attrib["x0"])) - rect[0] if (int(float(col.attrib["x0"])) - rect[0]) >0 else 0 )

                columns.sort()
                # print('column -> ',columns)
               
                for row in obj.findall(".//Row"):
                    rows.append(int(float(row.attrib["y0"])) - rect[1] if (int(float(row.attrib["y0"])) - rect[1]) >0 else 0)

                rows.sort()
                print('row -> ',rows)

                try:
                    columns_split = columns + [0,img_crop.shape[1]]
                    rows_split = rows + [0,img_crop.shape[0]]
                    columns_split.sort()
                    rows_split.sort()

                    columns_bg_colors = get_bg_color(img_crop,columns_split,axis=1)
                    rows_bg_colors,rows_text_colors = get_bg_and_text_color(img_crop,rows_split,axis=0,img_bw=img_crop_bw)
                    # print(rows_bg_colors)
                    # print(rows_text_colors)

                    colors_check= np.zeros_like(img_crop)
                    i=0
                    
                    column_bg =  max(set(columns_bg_colors), key=columns_bg_colors.count)
                    # print("COLUMN BG COLOR - ",column_bg)
                    for bg_col,text_col in zip(rows_bg_colors,rows_text_colors):
                        cv2.rectangle(colors_check,(10,10),(100,100),int(bg_col),-1)
                        cv2.rectangle(colors_check,(110,110),(210,210),int(text_col),-1)
                        
                        # if len(set(columns_bg_colors))>1:
                        #     print("COLUMNS COLOR CHANGINGGGGGG")
                        #     print(columns_bg_colors)
                        #     print(xml_file)
                        #     cv2.imshow('img',img_crop)
                        #     cv2.waitKey(0)
                        # elif text_col == columns_bg_colors[0]:
                        #     print("TEXT COLOR SAMMMMEEEEEEE")
                        #     print("ROW - ",i)
                        if text_col == column_bg and text_col!=bg_col:
                            # print("TEXT COLOR SAMMMMEEEEEEE")
                            # print("ROW - ",i)
                            # cv2.imshow('img',img_crop)
                            # cv2.waitKey(0)
                            new_color = int((int(bg_col)+int(text_col))/2)
                            # print(new_color)
                            rows_indices,cols_indices = np.where(img_crop_bw==text_col)
                            # print("ROWS_IND --- ",rows_indices)
                            # print("COLS_IND --- ",cols_indices)
                            for ind,row_i in enumerate(rows_indices):
                                    if row_i>=rows_split[i] and row_i<=rows_split[i+1]:
                                        table_copy[row_i][cols_indices[ind]]=new_color
                            # cv2.imshow("NEWCOL",table_copy)
                            # cv2.waitKey(0)
                        i+=1
                                
                            
                    # columns_colors = get_bg_color(img_crop,columns_split,axis=1)
                    # rows_colors = get_bg_color(img_crop,rows_split,axis=0)

                    
                    if len(set(rows_bg_colors))>1:
                        scale_column=0
                        for i in range(len(rows_split)-1):
                            bg_color_hgt = (rows_split[i+1]-rows_split[i])*rows_bg_colors[i]
                            scale_column+=bg_color_hgt
                    else:
                        scale_column= rows_bg_colors[0]*img_crop.shape[0]
                

                    scale_row = [row_color*img_crop.shape[1] for row_color in rows_bg_colors]

        

                    blank_row_indices = get_blank_rows_columns_indices(table_copy,1,rows_split,scale_row)
                    blank_col_indices = get_blank_rows_columns_indices(table_copy,0,columns_split,[scale_column])
                

                    col_gt_mask = np.zeros_like(img_crop)
                    row_gt_mask = np.zeros_like(img_crop)

                    all_white_rows = np.where(blank_row_indices>0.99)[0]
                    # print("rowssss\n",all_white_rows)

                    all_white_cols =np.where(blank_col_indices>0.98)[0]
                    # print("colssss\n",all_white_cols)
                    
                    prev_col_right_index = 0
                    for i,col in enumerate(columns):
                        # print(col)
                        # print('prev_col_right_index -> ',prev_col_right_index) 
                        col_index = np.where(all_white_cols==col)[0]
                        if col_index.size>0:
                            col_index = col_index[0]
                            # print(col_index)
        
                            left = get_left_or_above(all_white_cols[prev_col_right_index:col_index+1])  
                            right = get_right_or_below(all_white_cols[col_index:])

                            prev_col_right_index= np.where(all_white_cols==right)[0][0] +1
                        else :
                            left_neighbor,right_neighbor = get_neighbour_indices(col,all_white_cols)
                            # print('column neighbors -> ',left_neighbor,"  ",right_neighbor)
                            if left_neighbor.size>0:
                                left_neighbor_index = left_neighbor[0]
                                left_list = all_white_cols[prev_col_right_index:left_neighbor_index+1]
                                left= get_left_or_above(left_list)
                            else:
                                left = col

                            if right_neighbor.size>0:
                                right_neighbor_index = right_neighbor[0]
                                right_list = all_white_cols[right_neighbor_index:]
                                right= get_right_or_below(right_list)
                                prev_col_right_index= np.where(all_white_cols==right)[0][0] +1
                            else:
                                right = col

                        
                        if  i==0 and left-0<5:
                            left = col-5
                        elif i>0 and (left<=columns[i-1] or left-columns[i-1]<4):
                            left = col-3

                        if i==len(columns)-1  and img_crop.shape[1]-right <5:
                            right=col+10
                        elif i<len(columns)-1 and (right>=columns[i+1] or columns[i+1]-right<4):
                            right=col+3
                            prev_col_right_index= np.where(all_white_cols==right)[0][0] +1
                        
                                                
                        col_gt_mask[:, left : right+1] = 255

                    prev_row_below_index = 0
                    for i,row in enumerate(rows):
                        # print(row)
                        # print('prev_row_below_index -> ',prev_row_below_index)
                        row_index = np.where(all_white_rows==row)[0]
                        if row_index.size >0:

                            row_index=row_index[0]
                            # print("row index -> ",row_index)
                            above = get_left_or_above(all_white_rows[prev_row_below_index:row_index+1])  
                            below = get_right_or_below(all_white_rows[row_index:])
                            prev_row_below_index= np.where(all_white_rows==below)[0][0]+1
                        
                        else :
                            above_neighbor , below_neighbor = get_neighbour_indices(row,all_white_rows)
                            # print('row neighbors -> ',above_neighbor,"  ",below_neighbor)
                            if above_neighbor.size>0:
                                above_neighbor_index = above_neighbor[0]
                                above_list = all_white_rows[prev_row_below_index:above_neighbor_index+1]
                                above= get_left_or_above(above_list)
                            else:
                                above = row

                            if below_neighbor.size>0:
                                below_neighbor_index = below_neighbor[0]
                                below_list = all_white_rows[below_neighbor_index:]
                                below= get_right_or_below(below_list)
                                prev_row_below_index= np.where(all_white_rows==below)[0][0] +1
                            else:
                                below = row

                        if i==0 and above - 0<5:
                            above = row-5
                        elif i>0 and (above<=rows[i-1] or above-rows[i-1]<4):
                            above = row-3

                        if i==len(rows)-1  and img_crop.shape[0]-below < 5:
                            below = row+10
                        elif i<len(rows)-1 and (below>=rows[i+1] or rows[i+1]-below<4):
                            below=row+3
                            prev_row_below_index= np.where(all_white_rows==below)[0][0]+1  
                            
                    
                        row_gt_mask[above : below+1, :] = 255

                except:
                    continue


                cv2.imshow("colmns",col_gt_mask)
                cv2.waitKey(0)
                cv2.imshow("rows",row_gt_mask)
                cv2.waitKey(0)
                cv2.imwrite(
                    os.path.join(out_dir,"Masked_Documents", table_name + "_col.png"), col_gt_mask
                )
                cv2.imwrite(
                    os.path.join(out_dir,"Masked_Documents", table_name + "_row.png"), row_gt_mask
                )


                cv2.imwrite(
                    os.path.join(out_dir, "table_images", table_name + ".png"), img_crop
                )

                with open(
                    os.path.join(
                        out_dir, "table_split_labels", table_name + "_row.txt"
                    ),
                    "w",
                ) as f:
                    for i in row_gt_mask[:, 0]:
                        f.write(str(i) + "\n")

                with open(
                    os.path.join(
                        out_dir, "table_split_labels", table_name + "_col.txt"
                    ),
                    "w",
                ) as f:
                    for i in col_gt_mask[0, :]:
                        f.write(str(i) + "\n")



if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "-img",
        "--image_dir",
        type=str,
        help="Directory containing document-level images",
        default=r'/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/images',
        # required=True,
    )

    _parser.add_argument(
        "-xml",
        "--xml_dir",
        type=str,
        help="Directory containing document-level xmls",
        default=r'/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/correct_annotation_labels',
        # required=True,
    )

    _parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path of output directory for generated data",
        default=r'/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/out',
        # required=True,
    )

    args = _parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "Masked_Documents"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "table_images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "table_split_labels"), exist_ok=True)

    process_files(args.image_dir, args.xml_dir, args.out_dir)
