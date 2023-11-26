# from utility import open_image
# from pytesseract import Output, image_to_osd
from PIL import Image

class Validate:
    """
    Use all_validation() to perform all the validation check at once
    If not, the modules can also be called individually
    """
    def __init__(self, image):
        # image_path = image
        self.image = Image.open(image)
        # with open(image_path, 'rb') as f:
        #     data = f.read()
        # encoded_data = b64encode(data)
        # self.decoded_image = encoded_data.decode()
    

    def get_color_channel(self):
        return self.image.mode

    def check_corrupted(self):
        if self.image.size[0] > 0 and self.image.size[1] > 1:
            return False
        return True 
    
    def get_resolution(self):
        return (self.image.size[0], self.image.size[1])
    
    # def orientation_check(self):
    #     results = image_to_osd(self.image, output_type=Output.DICT)
    #     return results['orientation']
    
    
    def get_quality_score(self):
        return "Not Implemented"
    
    def all_validation(self):
        corrupted = self.check_corrupted()
        if not corrupted:
            result = {
                'corrupted' : 'No',
                'color_channel' : self.get_color_channel(),
                'resolution' : self.get_resolution(),
                # 'orientation' : self.orientation_check(),
                'quality_score' : self.get_quality_score()
            } 
            return result
        
        return 'Image Corrupted'