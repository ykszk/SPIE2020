import os
import numpy as np
from PIL import Image

def aligned_crop(in_dir, mask_dir, out_dir, ext):
    filenames = [fn for fn in os.listdir(in_dir) if os.path.splitext(fn)[1]==ext]

    bboxes = []
    ws, hs = [], []
    for fn in filenames:
        mask = np.array(Image.open(os.path.join(mask_dir,fn)))
        w = np.where(mask>0)
        xmin,xmax = w[1].min(), w[1].max()
        ymin,ymax = w[0].min(), w[0].max()
        bboxes.append((xmin,xmax,ymin,ymax))
        ws.append(xmax-xmin)
        hs.append(ymax-ymin)

    width = max(ws)
    height = max(hs)

    # %%
    p = 100 #padding size
    for fn, bbox, w, h in zip(filenames, bboxes, ws, hs):
        img = np.array(Image.open(os.path.join(in_dir,fn)))
        dw,dh = width-w, height-h
        xmin,xmax,ymin,ymax = np.array(bbox) + p
        padding_size = ((p,p),(p,p),(0,0)) if img.ndim == 3 else ((p,p),(p,p))
        img = np.pad(img, padding_size, mode='constant') # padding to prevent negative indices

        img = img[ymin-dh//2:ymax+dh//2+1,xmin-dw//2:xmax+dw//2+1]
        Image.fromarray(img).save(os.path.join(out_dir,fn))

import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cropping images with alignment.')
    parser.add_argument('input', help="Input directory",metavar='<input>')
    parser.add_argument('mask', help="Mask directory",metavar='<mask>')
    parser.add_argument('output', help="Output filename",metavar='<output>')
    parser.add_argument('--ext', help="Image file extension",metavar='<extension>',default='.png')

    args = parser.parse_args()
    aligned_crop(args.input, args.mask, args.output, args.ext)
