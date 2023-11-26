import PIL
from PIL import Image
import glob
import pandas as pd
import os
from hurry.filesize import size

  
image_directory="/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/PubTablesData/Merge_Prepared_Data/TableImages/"
resolution_dict={"ImageFileName":[],"Width":[],"Height":[],"Size":[],"Size in Bytes":[]}

with open('/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge/ImagesWithMoreThan5SecBackProp.txt') as file:
    image_files = [line.strip().replace(".png","") for line in file]
    
# for file in glob.glob(image_directory+"*"):
for file in image_files:
    file = os.path.join(image_directory,file+'.png')
    img = PIL.Image.open(file)
    # fetching the dimensions
    wid, hgt = img.size
    resolution_dict["ImageFileName"].append(file.split("/")[-1])
    resolution_dict["Width"].append(wid)
    resolution_dict["Height"].append(hgt)
    resolution_dict['Size'].append(size(os.path.getsize(file)))
    resolution_dict['Size in Bytes'].append(os.path.getsize(file))

resolution_df = pd.DataFrame(resolution_dict)
overall_resolution_info=open("/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge/Resolution_MoreThan5..txt","w")
overall_resolution_info.writelines("Minimum width and height = "+str(resolution_df['Width'].min())+"x"+str(resolution_df['Height'].min())+"\n")
overall_resolution_info.writelines("Maximum width and height = "+str(resolution_df['Width'].max())+"x"+str(resolution_df['Height'].max())+"\n")
overall_resolution_info.writelines("Median of width and height = "+str(resolution_df['Width'].median())+"x"+str(resolution_df['Height'].median())+"\n")
overall_resolution_info.writelines("Mean width and height = "+str(resolution_df['Width'].mean())+"x"+str(resolution_df['Height'].mean())+"\n")
overall_resolution_info.writelines("Minimum size = "+str(size(resolution_df['Size in Bytes'].min()))+"\n")
overall_resolution_info.writelines("Maximum size = "+str(size(resolution_df['Size in Bytes'].max()))+"\n")
overall_resolution_info.writelines("Median size = "+str(size(resolution_df['Size in Bytes'].median()))+"\n")
overall_resolution_info.writelines("Mean size = "+str(size(resolution_df['Size in Bytes'].mean()))+"\n")

overall_resolution_info.close()
resolution_df.to_csv("/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/Merge/Images_resolution_MoreThan5.csv")

    