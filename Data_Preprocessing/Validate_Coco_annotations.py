import os
import glob 
import json
from collections import defaultdict

images_directory='/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - test iii/Coco/images/'
annotations_directory='/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - test iii/Coco/annotations/'

print("Images in Directory: ",len(glob.glob(images_directory+"/**/*.png",recursive=True)))
c=0
annotated_images_id = defaultdict(list)
categories_id=[]
bbox_list = defaultdict(list)
table_category_id=None
for file in os.listdir(annotations_directory):
      with open(os.path.join(annotations_directory,file)) as json_file:
            annotations=json.load(json_file)
      for category in annotations['categories']:
            if category['name'].lower()=='table':
                  table_category_id=category['id']
      images_annotations= annotations['images']
      print("Images Mentioned in Annotations Json: ",len(images_annotations))
      areas_annotations=annotations['annotations']
      print('Number of Images Annotated: ',len(areas_annotations))
      for area_val in areas_annotations:
            if area_val['category_id']==table_category_id:
                  annotated_images_id[area_val['image_id']].append(area_val['area'])
                  bbox_list[area_val['image_id']].append(area_val['bbox'])
                  if area_val['area'] <=1500:
                        c+=1
            categories_id.append(area_val['category_id'])

print("Images with Table area less than 1500: ",c)
print("Categories : ",set(categories_id))
print('Table Category Id: ',table_category_id)
print("Unique Images which are Annotated: ",len(annotated_images_id.keys()))
print("Images with multiple Table Annotations: ")
for annot in annotated_images_id:
      if len(annotated_images_id[annot])>1:
            print(str(annot) +" : ",annotated_images_id[annot])



print("Images with multiple Tables' BBox: ")
for bbox in bbox_list:
      if len(bbox_list[bbox])>1:
            print(str(bbox)+" : ",bbox_list[bbox])


""" Result: 

Images in Directory:  298
Images Mentioned in Annotations Json:  298
Number of Images Annotated:  238
Images with Table area less than 1500:  0
Categories :  {1}
Table Category Id:  1
Unique Images which are Annotated:  238
Images with multiple Table Annotations: 
Images with multiple Tables' BBox: 

 """