# coding:utf-8
"""
convert png to jpg
"""

from PIL import Image
import glob
import re
import argparse

def main(args):
    original_path = args.originpath + args.generation
    convert_path  = args.convertpath + args.generation

    for f in glob.glob(original_path + '*.png'):
        img = Image.open(f)
        img = img.convert("RGBA")

        # make white back image
        conv = Image.new("RGB", img.size, (255,255,255))
        conv.paste(img, (0,0), img)

        f = f.replace(original_path, convert_path)
        print(f)

        f = f.replace('.png', '.jpg')
        print(f)

        conv.save(f)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='convert png to jpg image file')
    parser.add_argument('originpath', '-o', type=str, required=True)
    parser.add_argument('convertpath', '-t', type=str, required=True)
    parser.add_argument('generation', '-g', type=str, default='rs/', required=True)

    args = parser.parse_args()
    main(args)



