# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
from PIL import Image

class ImageSaver(object):
    def __init__(self, output_dir):
        self.counter = 0
        self.output_dir = output_dir

    def make_outputdir(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def save(self, im):
        im.save('%s/%03d.jpg' % (self.output_dir, self.counter))
        self.counter += 1

def crop(image, size):
    im = Image.open(image)
    w = im.width
    h = im.height
    p = (w - h) / 2
    box = (p, 0, p+h, h)
    return im.crop(box).resize((size, size))

def rotate_and_flip(im):
    for i in im, im.transpose(Image.ROTATE_90):
        yield i
        yield i.transpose(Image.FLIP_LEFT_RIGHT)
        yield i.transpose(Image.FLIP_TOP_BOTTOM)
        yield i.transpose(Image.ROTATE_180)

def main():
    parser = argparse.ArgumentParser(description='画像ファイル増やす君')
    parser.add_argument('--category', '-c', default='default', required=True,
        help='画像ファイルを格納しているディレクトリ名。raw/xxx の xxx の部分を指定してください。')
    args = parser.parse_args()

    # Initial values
    INPUT_DIR = './image/raw/%s' % (args.category)
    OUTPUT_DIR = './image/kakou/%s' % (args.category)
    CROP_SIZE = 28

    # Get images
    if not os.path.exists(INPUT_DIR):
        print('No such directory: %s' % (INPUT_DIR))
        sys.exit(1)
    images = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

    # Create instances for image save.
    ims = ImageSaver(OUTPUT_DIR)
    # Make directory for image save.
    ims.make_outputdir()

    # Increase each image and save it.
    for image in images:
        cropped = crop(image, CROP_SIZE)
        for arranged in rotate_and_flip(cropped):
            ims.save(arranged)

if __name__ == '__main__':
	  main()

