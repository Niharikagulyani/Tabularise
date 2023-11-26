import xml.etree.ElementTree as ET

class PubTablesFormat:
    def __init__(self,filename,path,database,width,height,depth):
        self.root = ET.Element('annotation')
        folder = ET.SubElement(self.root, 'folder')
        file_name = ET.SubElement(self.root, 'filename')
        file_name.text=filename
        Path = ET.SubElement(self.root, 'path')
        Path.text=path
        source = ET.SubElement(self.root,'source')
        Database = ET.SubElement(source,'database')
        Database.text=database
        size = ET.SubElement(self.root,'size')
        Width=ET.SubElement(size,'width')
        Width.text=width
        Height=ET.SubElement(size,'height')
        Height.text=height
        Depth=ET.SubElement(size,'depth')
        Depth.text=depth
        segmented=ET.SubElement(self.root,'segmented')
        segmented.text=0
    
    def add_object(self,name,xmin_val,ymin_val,xmax_val,ymax_val):
        object=ET.SubElement(self.root,'object')
        object_name=ET.SubElement(object,'name')
        object_name.text=name
        pose=ET.SubElement(object,'pose')
        pose.text='Frontal'
        truncated=ET.SubElement(object,'truncated')
        truncated.text='0'
        difficult=ET.SubElement(object,'difficult')
        difficult.text='0'
        occluded=ET.SubElement(object,'occluded')
        occluded.text='0'
        bndbox=ET.SubElement(object,'bndbox')
        xmin=ET.SubElement(bndbox,'xmin')
        xmin.text=xmin_val
        ymin=ET.SubElement(bndbox,'ymin')
        ymin.text=ymin_val
        xmax=ET.SubElement(bndbox,'xmax')
        xmax.text=xmax_val
        ymax=ET.SubElement(bndbox,'ymax')
        ymax.text=ymax_val
    
    def save_xml(self,output_path):
        tree = ET.ElementTree(self.root)
        tree.write(output_path)
        