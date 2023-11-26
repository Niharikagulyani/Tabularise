import os
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--merges_dir", type=str,
                    help="Directory for Merge Data", default="/home/ntlpt-52/work/IDP/Table_Extraction/Training_Code/PubTablesData/OutputFromMerge/merges")

args = parser.parse_args()
merges_dir = args.merges_dir

empty_merge_file = []

for file in os.listdir(merges_dir):
    # objects = []
    with (open(os.path.join(merges_dir,file), "rb")) as openfile:
        # while True:
        #     try:
        #         objects.append(pickle.load(openfile))
        #     except EOFError:
        #         break
        objects = pickle.load(openfile)
    if len(objects['row'])==0 and len(objects['col'])==0:
        # print(objects['row'])
        # print(objects['col'])
        # print(file)
        empty_merge_file.append(file)
        print(file)
    # print(objects)
    # exit()