import os
import glob
import json
import itertools
import pandas as pd
from collections import defaultdict
from xml.etree import ElementTree
import shutil
import cv2

class AnnotationFiles:
    def __init__(self,xml_directories,images_directories):
        self.xml_directories=xml_directories
        self.images_directories=images_directories
        self.images_annotations_done=[]
        self.single_table_annotations=defaultdict(list)
        self.multiple_table_annotations=defaultdict(list)
        self.images_annotations=defaultdict(dict)
        self.train_annotations={
            "licenses": [
                {
                    "name": "",
                    "id": 0,
                    "url": ""
                }
            ],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            "categories": [
                {
                    "id": 1,
                    "name": "Table",
                    "supercategory": ""
                },
                {
                    "id": 2,
                    "name": "Rows",
                    "supercategory": ""
                },
                {
                    "id": 3,
                    "name": "Columns",
                    "supercategory": ""
                },
                {
                    "id": 4,
                    "name": "merged cell",
                    "supercategory": ""
                }
            ],
            "images":[],
            "annotations": []

        }

        self.test_annotations={
            "licenses": [
                {
                    "name": "",
                    "id": 0,
                    "url": ""
                }
            ],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            "categories": [
                {
                    "id": 1,
                    "name": "Table",
                    "supercategory": ""
                },
                {
                    "id": 2,
                    "name": "Rows",
                    "supercategory": ""
                },
                {
                    "id": 3,
                    "name": "Columns",
                    "supercategory": ""
                },
                {
                    "id": 4,
                    "name": "merged cell",
                    "supercategory": ""
                }
            ],
            "images":[],
            "annotations": []
        }

        
    
    def get_single_multi_Annotations(self):
        single_table_annots=defaultdict(list)
        multi_table_annots = defaultdict(list)
        images_info=defaultdict(dict)
        mismatch_count=0
        mismatching_Img_annots_tobedeleted=[]

        for xml_dir , img_dir in zip(self.xml_directories,self.images_directories):
            for xml_file in os.listdir(xml_dir):
                img_file_name=xml_file.split(".")[0]+'.png'
                tree = ElementTree.parse(os.path.join(xml_dir,xml_file))
                root = tree.getroot()
                width=eval(root.attrib['width'])
                height=eval(root.attrib['height'])
                tables= root.findall(".//Table")
                if not tables:
                    continue
                img_annotations = []
                for i, obj in enumerate(tables):
                    table_rect = [
                                (eval(obj.attrib["x0"])),
                                (eval(obj.attrib["y0"])),
                                (eval(obj.attrib["x1"])),
                                (eval(obj.attrib["y1"])),
                            ]
                    if table_rect[2]>table_rect[0] and table_rect[3]>table_rect[1]:
                        img_annotations.append(table_rect)
                    
                if len(img_annotations)==0:
                    continue

                #___________________________________MISMATCHING OF IMAGE/TABLE ANNOTATIONS CHECK__________________________________________
                if img_file_name in self.images_annotations_done:

                    if width!=images_info[img_file_name]['width'] or height!=images_info[img_file_name]['height']:
                        print("MISMATCH IN HEIGHT AND WIDTH FOUND IN IMAGE -> ",img_file_name)

                    if len(img_annotations)>1 and img_file_name in single_table_annots.keys():
                        print("MISMATCH FOUND -> ", img_file_name," HAS MULTIPLE ANNOTATIONS ")
                        print("BUT SINGLE Annotations found before")
                        mismatch_count+=1
                        # print("Table Annotations Before ..........",single_table_annots[img_file_name])
                        # print("Table Annotations Now ..........",img_annotations)
                        # if not single_table_annots[img_file_name][0] in img_annotations:
                        #     print("NO COMMON ANNOTATIONS")
                        # else :
                        #     print("COMMON ANNOTATIONS")

                    elif len(img_annotations)>1:
                        if img_annotations!=multi_table_annots[img_file_name]:
                            print("MISMATCH FOUND IN BOTH MULTI ANNOTATIONS -> ",img_file_name)
                            mismatch_count+=1
                        else:
                            print("SAME MULTI ANNOTATIONS AS BEFORE -> ",img_file_name)
                        # print("Table Annotations Before ..........",multi_table_annots[img_file_name])
                        # print("Table Annotations Now ..........",img_annotations)


                    elif len(img_annotations)==1 and img_file_name in multi_table_annots.keys():
                        print("MISMATCH FOUND -> ",img_file_name," HAS SINGLE ANNOTATIONS")
                        print("BUT MULTIPLE ANNOTATIONS FOUND BEFORE")
                        mismatch_count+=1
                        # print("Table Annotations Before ..........", multi_table_annots[img_file_name])
                        # print("Table Annotations Now ..........",img_annotations)
                        # if not img_annotations[0] in multi_table_annots[img_file_name]:
                        #     print("NO COMMON ANNOTATIONS")
                        # else :
                        #     print("COMMON ANNOTATIONS")

                    else:
                        if img_annotations!=single_table_annots[img_file_name]:
                            # print("MISMATCH FOUND IN BOTH SINGLE ANNOTATIONS -> ",img_file_name)
                            mismatch_count+=1
                            del_flag=False

                            """ 

                            THERE IS MISMATCHING IN TABLE ANNOTATIONS IN DUPLICATED XML FILES
                            SO FOR SINGLE TABLE IMAGES:
                                    IF DIFFERENCE BETWEEN X,Y,WIDTH OR HEIGHT > 50 , WE'LL REJECT THE IMAGE
                                    ELSE KEEP ANY ONE ANNOTATIONS
                            
                            BUT SOME IMAGES HAVE MORE THAN 2 DIFFERENT ANNOTATIONS 
                            AND THERE CAN BE 1 ANNOTATION WITH DIFFERENCE LESS THAN 50 
                            SO ACCEPTING IMAGE IN THIS CASE

                            """

                            
                            for coord,annot in zip(img_annotations[0],single_table_annots[img_file_name][0]):
                                if abs(coord-annot)>50:
                                    if img_file_name not in mismatching_Img_annots_tobedeleted:
                                        mismatching_Img_annots_tobedeleted.append(img_file_name)
                                    del_flag=True

                            if del_flag==False and img_file_name in mismatching_Img_annots_tobedeleted:
                                mismatching_Img_annots_tobedeleted.remove(img_file_name)

                        # else:
                        #     print("SAME SINGLE ANNOTATIONS AS BEFORE -> ",img_file_name)
                        # print("Table Annotations Before ..........",single_table_annots[img_file_name])
                        # print("Table Annotations Now ..........",img_annotations)

                else:
                    if len(img_annotations)>1:
                        multi_table_annots[img_file_name].extend(img_annotations)
                    else:
                        single_table_annots[img_file_name].extend(img_annotations)
                    self.images_annotations_done.append(img_file_name)
                    images_info[img_file_name]={'width':width,'height':height,'imgpath':os.path.join(img_dir,img_file_name)}
                    if not os.path.exists(os.path.join(img_dir,img_file_name)):
                        print("IMAGE DOESN'T EXISTS")
                        print(os.path.join(img_dir,img_file_name))

        for img_file in mismatching_Img_annots_tobedeleted:
            del single_table_annots[img_file]
        
        self.single_table_annotations=single_table_annots
        self.multiple_table_annotations=multi_table_annots
        self.images_annotations=images_info

        return single_table_annots,multi_table_annots

    
    def get_images_with_no_tables(self):
        images_with_no_table=[]
        for xml_dir in self.xml_directories:
            for xml_file in os.listdir(xml_dir):
                img_file_name=xml_file.split(".")[0]+'.png'
                tree = ElementTree.parse(os.path.join(xml_dir,xml_file))
                root = tree.getroot()
                tables= root.findall(".//Table")
                if not tables:
                    images_with_no_table.append(img_file_name)
        
        return images_with_no_table

    def get_images_with_wrong_table_coordinates(self):
        images_with_wrong_tables=[]
        for xml_dir in self.xml_directories:
            for xml_file in os.listdir(xml_dir):
                img_file_name=xml_file.split(".")[0]+'.png'
                tree = ElementTree.parse(os.path.join(xml_dir,xml_file))
                root = tree.getroot()
                tables= root.findall(".//Table")
                if not tables:
                    continue
                table_annotations = []
                for obj in tables:
                    table_rect = [
                                (eval(obj.attrib["x0"])),
                                (eval(obj.attrib["y0"])),
                                (eval(obj.attrib["x1"])),
                                (eval(obj.attrib["y1"])),
                            ]
                    if table_rect[2]>table_rect[0] and table_rect[3]>table_rect[1]:
                        table_annotations.append(table_rect)
                    
                if len(table_annotations)==0:
                    images_with_wrong_tables.append(img_file_name)

        return images_with_wrong_tables
    
    def save_train_test_json_annotations(self,train_destination,test_destination):

        os.makedirs(os.path.join(train_destination,"Images"),exist_ok=True)
        os.makedirs(os.path.join(train_destination,"Annotations"),exist_ok=True)
        os.makedirs(os.path.join(test_destination,"Images"),exist_ok=True)
        os.makedirs(os.path.join(test_destination,"Annotations"),exist_ok=True)


        #_________________To keep 80% of single table and multi table annotations in train_data____________________________________
        single_train_len = int(0.8*len(self.single_table_annotations.keys()))
        multi_train_len=int(0.8*len(self.multiple_table_annotations.keys()))
        single_test_len= len(self.single_table_annotations.keys())-single_train_len
        multi_test_len= len(self.multiple_table_annotations.keys())-multi_train_len
        train_img_id=1
        train_annot_id=1
        test_img_id=1
        test_annot_id=1
        #__________________________________ADDING SINGLE TABLE ANNOTATIONS FIRST___________________________________________
        for i,img_file_name in enumerate(self.single_table_annotations.keys()):
            if i<single_train_len:
                image_info= {
                    "id": train_img_id,
                    "width": self.images_annotations[img_file_name]['width'],
                    "height": self.images_annotations[img_file_name]['height'],
                    "file_name": img_file_name,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                }
                shutil.copy(self.images_annotations[img_file_name]['imgpath'],os.path.join(train_destination,"Images",img_file_name))
                self.train_annotations["images"].append(image_info)
                table_coords = self.single_table_annotations[img_file_name][0]
                bbox = [table_coords[0],table_coords[1],table_coords[2]-table_coords[0],table_coords[3]-table_coords[1]]

                table_annotations={
                    "id": train_annot_id,
                    "image_id": train_img_id,
                    "category_id": 1,
                    "segmentation": [],
                    "area": bbox[2]*bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False
                    }
                }
                self.train_annotations["annotations"].append(table_annotations)
                train_img_id+=1
                train_annot_id+=1
            else:
                image_info= {
                    "id": test_img_id,
                    "width": self.images_annotations[img_file_name]['width'],
                    "height": self.images_annotations[img_file_name]['height'],
                    "file_name": img_file_name,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                }
                shutil.copy(self.images_annotations[img_file_name]['imgpath'],os.path.join(test_destination,"Images",img_file_name))
                self.test_annotations["images"].append(image_info)
                table_coords = self.single_table_annotations[img_file_name][0]
                bbox = [table_coords[0],table_coords[1],table_coords[2]-table_coords[0],table_coords[3]-table_coords[1]]

                table_annotations={
                    "id": test_annot_id,
                    "image_id": test_img_id,
                    "category_id": 1,
                    "segmentation": [],
                    "area": bbox[2]*bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False
                    }
                }
                self.test_annotations["annotations"].append(table_annotations)
                test_img_id+=1
                test_annot_id+=1

        #___________________________________________ADDING MULTIPLE TABLE ANNOTATIONS_________________________________________________
        for i,img_file_name in enumerate(self.multiple_table_annotations.keys()):
            if i<multi_train_len:
                image_info= {
                    "id": train_img_id,
                    "width": self.images_annotations[img_file_name]['width'],
                    "height": self.images_annotations[img_file_name]['height'],
                    "file_name": img_file_name,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                }
                shutil.copy(self.images_annotations[img_file_name]['imgpath'],os.path.join(train_destination,"Images",img_file_name))
                self.train_annotations["images"].append(image_info)
                for coords in self.multiple_table_annotations[img_file_name]:
                    table_coords = coords
                    bbox = [table_coords[0],table_coords[1],table_coords[2]-table_coords[0],table_coords[3]-table_coords[1]]

                    table_annotations={
                        "id": train_annot_id,
                        "image_id": train_img_id,
                        "category_id": 1,
                        "segmentation": [],
                        "area": bbox[2]*bbox[3],
                        "bbox": bbox,
                        "iscrowd": 0,
                        "attributes": {
                            "occluded": False
                        }
                    }
                    train_annot_id+=1
                    self.train_annotations["annotations"].append(table_annotations)
                
                train_img_id+=1
                
            else:
                image_info= {
                    "id": test_img_id,
                    "width": self.images_annotations[img_file_name]['width'],
                    "height": self.images_annotations[img_file_name]['height'],
                    "file_name": img_file_name,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0
                }
                shutil.copy(self.images_annotations[img_file_name]['imgpath'],os.path.join(test_destination,"Images",img_file_name))
                self.test_annotations["images"].append(image_info)
                for coords in self.multiple_table_annotations[img_file_name]:
                    table_coords = coords
                    bbox = [table_coords[0],table_coords[1],table_coords[2]-table_coords[0],table_coords[3]-table_coords[1]]

                    table_annotations={
                        "id": test_annot_id,
                        "image_id": test_img_id,
                        "category_id": 1,
                        "segmentation": [],
                        "area": bbox[2]*bbox[3],
                        "bbox": bbox,
                        "iscrowd": 0,
                        "attributes": {
                            "occluded": False
                        }
                    }
                    test_annot_id+=1
                    self.test_annotations["annotations"].append(table_annotations)
                
                test_img_id+=1

        with open(os.path.join(train_destination,"Annotations","TrainAnnotations.json"), "w") as outfile:
            outfile.write(json.dumps(self.train_annotations))

        with open(os.path.join(test_destination,"Annotations","TestAnnotations.json"), "w") as outfile:
            outfile.write(json.dumps(self.test_annotations))




if __name__ == "__main__":
    train_destination="/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/TableDetectionData/Training_test"
    test_destination='/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/TableDetectionData/Testing_test'
    xml_dirs=['/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table-training/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table training II/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - training iii/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table-test/T-Truth/labels','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - test iii/T-Truth/labels']
    images_dirs=['/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table-training/T-Truth/images','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table training II/T-Truth/images','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - training iii/T-Truth/images','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/images','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table-test/T-Truth/images','/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/table - test iii/T-Truth/images']
    annotations=AnnotationFiles(xml_dirs,images_dirs)
    wrong_table_imgs=annotations.get_images_with_wrong_table_coordinates()
    no_table_imgs=annotations.get_images_with_no_tables()
    single_annots,mult_annots=annotations.get_single_multi_Annotations()
    Train_Test=annotations.save_train_test_json_annotations(train_destination=train_destination,test_destination=test_destination)
    print("LENGTH OF WRONG TABLE IMAGES: ",len(wrong_table_imgs))
    print("LENGTH OF NO TABLE IMAGES: ",len(no_table_imgs))
    print("LENGTH OF SINGLE ANNOTATIONS: ",len(single_annots))
    print("LENGTH OF MULTIPLE ANNOTATIONS: ",len(mult_annots))


