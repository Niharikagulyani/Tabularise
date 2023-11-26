import os
from CorruptedFIles import Validate
count = 0
for root_dir, cur_dir, files in os.walk("/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3"):

    for file in files:
        if file.endswith(".png"):
            file_validation=Validate(os.path.join(root_dir,file)).all_validation()
            if file_validation=='Image Corrupted':
                print('Image Corrupted',file)
                exit()
            count+=1


print('Images in Trainingsetfinal3 complete-> file count:', count)
print("images in Trainingsetfinal3  t-truth : ",len(os.listdir("/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/images")))
print("Labels in Trainingsetfinal3  t-truth: ",len(os.listdir('/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/labels')))

for file in os.listdir('/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/images'):
    if os.path.exists("/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/labels/"+file.split(".")[0]+".xml"):
        with open("/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/labels/"+file.split(".")[0]+".xml", 'r') as f:
            data = f.read()
        if not data.strip():
            print("XML Empty","/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/labels/"+file+".xml")
    else:
        print("/home/ntlpt-52/work/IDP/Table_Extraction/latest_data_Mar16_2023/trainingsetfinal3/T-Truth/labels/"+file.split(".")[0]+".xml")
        print("NO XML PRESENT FOR ",file)


