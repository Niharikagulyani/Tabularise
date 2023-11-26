import glob 

labels_dir ='/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/Prepared_Data/table_split_labels'

files_with_wrong_labels = []

for label_file in glob.glob(labels_dir+"/*"):
    with open(label_file) as file:
        labels = [line.strip() for line in file]
    if len(set(labels)) !=2:
        files_with_wrong_labels.append(label_file.split("/")[-1])

with open("Files_With_Wrong_Labels.txt",'w') as file:
    for f in files_with_wrong_labels:
        file.write(f)
        file.write('\n')