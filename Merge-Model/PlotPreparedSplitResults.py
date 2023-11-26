import os
import pickle 
import argparse
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import utils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images_dir", help="Path to images.", default="/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/Prepared_Data/table_images")
parser.add_argument("-s", "--split_outs_dir", help="Path to Split model output.", default="/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/Prepared_Data/table_split_labels")

args = parser.parse_args()
split_outs_dir = args.split_outs_dir
images_dir = args.images_dir

# save_plots='/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge/Merge_Model/split_outs_old'
# os.makedirs(save_plots,exist_ok=True)

for file in os.listdir(split_outs_dir):
    print(file)
    with open(os.path.join(split_outs_dir,file), "rb") as openfile:
        objects = pickle.load(openfile)
    # with open(os.path.join('/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge/Merge_Model/SPLIT_RESULTS/split_outs_Pub50',file), "rb") as openfile:
    #     probabs = pickle.load(openfile)
    image = cv2.imread(os.path.join(images_dir,file.replace('pkl','png')))
    H, W, C = image.shape
    image_trans = image.transpose((2, 0, 1)).astype("float32")
    resized_image = utils.resize_image(image_trans)
    input_image = utils.normalize_numpy_image(resized_image).unsqueeze(0)
    row_prob = objects["row_prob"]
    col_prob = objects["col_prob"]
    thresh = 0.7
    cpn_image = utils.probs_to_image(col_prob.detach().clone(), input_image.shape, axis=0)
    rpn_image = utils.probs_to_image(row_prob.detach().clone(), input_image.shape, axis=1)
    grid_img,row_image, col_image = utils.binary_grid_from_prob_images(
				rpn_image, cpn_image
	)
    grid_np_img = utils.tensor_to_numpy_image(grid_img)
    row_np_image = utils.tensor_to_numpy_image(row_image)
    col_np_image = utils.tensor_to_numpy_image(col_image)
    # print(np.unique(row_np_image))
    # print(np.unique(col_np_image))
    # if len(np.unique(row_np_image)) ==1:
    #     print("ROW EMPTY")
    # if len(np.unique(col_np_image)) ==1:
    #     print("COL EMPTY")
    


    plt.imshow(row_np_image)
    plt.show()
    # plt.savefig(save_plots+file.replace(".pkl","")+"_rows.png")
    plt.imshow(col_np_image)
    plt.show()
    # plt.savefig(save_plots+file.replace(".pkl","")+"_cols.png")

    plt.imshow(grid_np_img, interpolation='nearest')
    plt.show()

    # row_p=probabs['row_prob']
    # col_p=probabs['col_prob']
    # cpn_image_k = utils.probs_to_image(col_p.detach().clone(), input_image.shape, axis=0)
    # rpn_image_k = utils.probs_to_image(row_p.detach().clone(), input_image.shape, axis=1)
    # grid_img,row_image, col_image = utils.binary_grid_from_prob_images(
	# 			rpn_image, cpn_image
	# )
    # row_image_k, col_image_k = utils.binary_grid_from_prob_images(
	# 			rpn_image_k, cpn_image_k
	# )

    # # grid_np_img = utils.tensor_to_numpy_image(grid_img)
    # row_np_image_k = utils.tensor_to_numpy_image(row_image_k)
    # col_np_image_k = utils.tensor_to_numpy_image(col_image_k)

    # fig = plt.figure(figsize=(10, 7))
    # rows,columns=2,2
    # fig.add_subplot(rows,columns,1)
    # plt.imshow(row_np_image)
    # fig.add_subplot(rows,columns,2)
    # plt.imshow(row_np_image_k)
    # fig.add_subplot(rows,columns,3)
    # plt.imshow(col_np_image)
    # fig.add_subplot(rows,columns,4)
    # plt.imshow(col_np_image_k)
    # plt.show()

