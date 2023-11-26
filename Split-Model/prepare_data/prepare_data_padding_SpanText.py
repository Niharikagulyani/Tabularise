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


def get_bbox(element)-> list:
    bounding_box = element.find('bndbox')
    xmin=int(eval(bounding_box.find('xmin').text))
    ymin=int(eval(bounding_box.find('ymin').text))
    xmax=int(eval(bounding_box.find('xmax').text))
    ymax=int(eval(bounding_box.find('ymax').text))

    return [xmin,ymin,xmax,ymax]

def plot_indices(img,indices,axis):
    mask_image = np.zeros_like(img)
    height,width =img.shape 
    if axis==0:
        for i in indices:
            cv2.line(mask_image,(0,i),(width,i),255,2)
    elif axis ==1:
        for i in indices:
            cv2.line(mask_image,(i,0),(i,height),255,2)
    
    return mask_image


def get_pixels_percentage_respect_to_bg(img,axis,bg_color):
    if bg_color == 0:
        img=255-img
        bg_color = 255
    shape = img.shape
    sum_of_pixels = np.sum(img,axis=axis)
    scale = shape[axis]
    white_pixels_percentage = sum_of_pixels/(bg_color*scale)

    return white_pixels_percentage

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


def get_bg_and_text_color(img_bw):
        
    unique_bw, counts_bw = np.unique(img_bw, return_counts=True)
    text_color = unique_bw[int(np.argmin(counts_bw))]
    bg_color =  unique_bw[int(np.argmax(counts_bw))]

    return bg_color,text_color

def bw_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img_bin = 255 - thresh
    # thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    return thresh,img_bin

def remove_lines(thresh,img_bin_inv):

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img_bin_inv, [c], -1, (255,255,255), 2) # To remove horizontal lines in original document
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img_bin_inv, [c], -1, (255,255,255), 2) # To remove vertical lines in original document  

    return img_bin_inv


def get_padding(blank_indices,row_col_indices,img,axis):
    # print('in padding indices- ',row_col_indices)
    height,width = img.shape
    if axis==0:
        last_index= height
    else:
        last_index = width
    mask_image = np.zeros_like(img)


    last_padded_id = 0
    # print('starting loop')
    for i,row_col_id in enumerate(row_col_indices):
        # print(i, ' id - ' ,row_col_id)
        index = np.where(blank_indices==row_col_id)[0]
        if index.size>0:
            index=index[0]
            left_above = get_left_or_above(blank_indices[last_padded_id:index+1])  
            right_down = get_right_or_below(blank_indices[index:])
            # print('1. left_above - ',left_above)
            # print('1. right_down - ',right_down)
            last_padded_id= np.where(blank_indices==right_down)[0][0] +1
        else:
            left_neighbor,right_neighbor = get_neighbour_indices(row_col_id,blank_indices)
            if left_neighbor.size>0:
                left_neighbor_index = left_neighbor[0]
                left_list = blank_indices[last_padded_id:left_neighbor_index+1]
                left_above= get_left_or_above(left_list)
            else:
                left_above = row_col_id

            if right_neighbor.size>0:
                right_neighbor_index = right_neighbor[0]
                right_list = blank_indices[right_neighbor_index:]
                right_down= get_right_or_below(right_list)
                # print('2. left_above - ',left_above)
                # print('2. right_down - ',right_down)
                last_padded_id= np.where(blank_indices==right_down)[0][0] +1
            else:
                right_down = row_col_id
        
        # print('3. left_above - ',left_above)
        # print('3. right_down - ',right_down)
        if  i==0 and left_above-0<5 and row_col_id!=0:
            left_above = row_col_id-3
        elif i>0 and left_above-row_col_indices[i-1]<4:
            left_above = row_col_id-3

        if i==len(row_col_indices)-1  and last_index-right_down <5:
            right_down=row_col_id+5
        elif i<len(row_col_indices)-1 and right_down-row_col_indices[i+1]>-4:
            try:
                right_down=row_col_id+3
                last_padded_id= np.where(blank_indices==right_down)[0][0] +1
            except:
                print('RD',right_down)
                print('LAST PADDED ID ',last_padded_id)
                print("ID - ",row_col_id)
                print("TOTAL id - ",row_col_indices)
                print('list blank - ',blank_indices)

        # print('4. left_above - ',left_above)
        # print('4. right_down - ',right_down)

        
        
        if axis ==0:
            mask_image[left_above : right_down+1, :] = 255
            
        elif axis == 1 :                      
            mask_image[:, left_above : right_down+1] = 255

    return mask_image

def get_row_col_indices(row_objects,col_objects,table_coords):
    table_x0 = table_coords[0]
    table_y0 = table_coords[1]
    table_x1 = table_coords[2]
    table_y1 = table_coords[3]
    row_indices=[]
    col_indices=[]
    for row_obj in row_objects:
        bounding_box = row_obj.find('bndbox')
        ymax=int(eval(bounding_box.find('ymax').text))
        if ymax-table_y1>-4:
            continue
        row_indices.append(ymax-table_y0)
    for col_obj in col_objects:
        bounding_box = col_obj.find('bndbox')
        xmax=int(eval(bounding_box.find('xmax').text))
        if xmax-table_x1>-4:
            continue
        col_indices.append(xmax-table_x0)

    row_indices.sort()
    col_indices.sort()
    return row_indices,col_indices

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

    print('++++++++++++',len(files))
    files.sort()

    for ii, file in enumerate(files):
       
        filename = file
        image_file = os.path.join(image_dir, filename + ".jpg")
        xml_file = os.path.join(xml_dir, filename + ".xml")

        img = cv2.imread(image_file)

        if (
            os.path.exists(image_file)
            and os.path.exists(xml_file)
        ):
            print("[", ii, "/", len(files), "]", "Processing: ", file)
            tree = ElementTree.parse(xml_file)
            root = tree.getroot()
            tables= root.findall("./object/[name='table']")
            if len(tables)>1:
                continue
        
            table_name = filename + "_0"
            
            table = tables[0]
            rect = get_bbox(table)
    
            img_crop = img[rect[1] : rect[3], rect[0] : rect[2]]
            thresh,img_crop_bw = bw_img(img_crop)
            table_copy = img_crop.copy()

            all_cols_objects = root.findall("./object/[name='table column']")
            all_rows_objects = root.findall("./object/[name='table row']")
            spanning_cells = root.findall("./object/[name='table spanning cell']")

            row_indices,col_indices = get_row_col_indices(all_rows_objects,all_cols_objects,rect)


            try:
                column_header= root.findall("./object/[name='table column header']")
                column_header = column_header[0]
            except:
                print('column header not found')
                continue
            
            column_header_coords= get_bbox(column_header)
            print('rect -',rect)
            print('column_header_coords -',column_header_coords)
            ch = [column_header_coords[0]-rect[0],column_header_coords[1]-rect[1],column_header_coords[2]-rect[0],column_header_coords[3]-rect[1]]
            ch =[c if c>0 else 0 for c in ch]
            print('ch -',ch)
            column_header_bg_color , column_header_text_color  = get_bg_and_text_color(img_crop_bw[ch[1]:ch[3],ch[0]:ch[2]])
            if column_header_bg_color ==0:
                img_crop_bw[ch[1]:ch[3],ch[0]:ch[2]] = 255 - img_crop_bw[ch[1]:ch[3],ch[0]:ch[2]]

            img_crop_bw= remove_lines(thresh,img_crop_bw)

            for cell in spanning_cells:
                cell_bbox = get_bbox(cell)
                
                x0 = cell_bbox[0]-rect[0]
                y0 = cell_bbox[1]-rect[1]
                x1 = cell_bbox[2]-rect[0]
                y1 = cell_bbox[3]-rect[1]
                cv2.rectangle(table_copy,(x0,y0),(x1,y1),0,4)
                img_crop_bw[y0:y1,x0:x1]=255
            

            # cv2.imshow('span cells',table_copy)
            # cv2.waitKey(0)
            # cv2.imshow('img_crop_bw',img_crop_bw)
            # cv2.waitKey(0)

            rows_perc_table = get_pixels_percentage_respect_to_bg(img_crop_bw,axis=1,bg_color=255)
            cols_perc_table = get_pixels_percentage_respect_to_bg(img_crop_bw,axis=0,bg_color=255)

            blank_rows_table = np.where(rows_perc_table>0.99)[0]
            blank_cols_table = np.where(cols_perc_table>0.99)[0]

            row_padding_img= get_padding(blank_rows_table,row_indices,img_crop_bw,axis=0)
            col_padding_img= get_padding(blank_cols_table,col_indices,img_crop_bw,axis=1)


            # cv2.imshow('col padding',col_padding_img)
            # cv2.waitKey(0)
            # cv2.imshow('row padding',row_padding_img)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(out_dir,'Masked_Documents',filename+"_col.png"),col_padding_img)
            cv2.imwrite(os.path.join(out_dir,'Masked_Documents',filename+"_row.png"),row_padding_img)

            cv2.imwrite(
                os.path.join(out_dir, "table_images", table_name + ".png"), img_crop
            )

            with open(
                os.path.join(
                    out_dir, "table_split_labels", table_name + "_row.txt"
                ),
                "w",
            ) as f:
                for i in row_padding_img[:, 0]:
                    f.write(str(i) + "\n")

            with open(
                os.path.join(
                    out_dir, "table_split_labels", table_name + "_col.txt"
                ),
                "w",
            ) as f:
                for i in col_padding_img[0, :]:
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
        default=r'/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/Pascal_VOC_labels',
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
