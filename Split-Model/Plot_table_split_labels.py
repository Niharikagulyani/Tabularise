import cv2
import os 
# import xml.etree.ElementTree as ET
import numpy as np
# import shutil
from matplotlib import pyplot as plt

labels_dir = '/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/RowColumSplit/PubTables_PreparedData/table_split_labels'
images_dir = '/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/RowColumSplit/PubTables_PreparedData/table_images'

count =0

for img_file in os.listdir(images_dir):
    if count ==5:
        break
    img = cv2.imread(os.path.join(images_dir,img_file))
    print(img_file)
    # print(img.shape)
    col_file = img_file.split(".")[0]+'_col.txt'
    row_file = img_file.split(".")[0]+'_row.txt'
    with open(os.path.join(labels_dir,col_file)) as file:
        columns = np.array([line.strip() for line in file])
    with open(os.path.join(labels_dir,row_file)) as file:
        rows = np.array([line.strip() for line in file])
    # col_img = np.zeros_like((img.shape[0],img.shape[1]))
    # row_img = np.zeros_like((img.shape[0],img.shape[1]))
    col_img = np.tile(columns,(img.shape[0],1)).astype(np.float)
    rows=np.repeat(rows,img.shape[1])
    row_img = rows.reshape(img.shape[0],img.shape[1]).astype(np.float)
    # print(col_img.shape)
    # print(col_img,'\n\n')
    # print(row_img.shape)
    # print(row_img,'\n\n')
    plt.imshow(col_img, interpolation='nearest')
    plt.show()
    plt.imshow(row_img, interpolation='nearest')
    plt.show()
    count+=1
