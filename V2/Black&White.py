from skimage.io import imread,imsave, imshow
from skimage import img_as_float
import skimage
import numpy
from PIL import Image
import os

class DataImage():
    def __init__(self,input_image_path):
        self.im_float = ""
        self.input_image_path = input_image_path

    def resize_image(self):
        input_image_path = self.input_image_path
        size = (35, 35)
        output_image_path = "asx.jpg"
        original_image = Image.open(input_image_path)
        width, height = original_image.size


        resized_image = original_image.resize(size)
        width, height = resized_image.size

        resized_image.save(output_image_path)
        self.output = output_image_path

    def image(self):
        numpy.set_printoptions(threshold=numpy.nan)
        im = imread(self.output)
        im_float = img_as_float(im)
        im_float = numpy.asarray(skimage.color.rgb2grey(im_float))
        imsave("123.png",im_float)

        return im_float.reshape(-1)

k = DataImage("/media/sf_shared/MAILHACK/train/-1 other/0f76ae42831276d5b3148d1a7b772df1.jpg")
k.resize_image()
print(k.image())
