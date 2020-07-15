import os
import itertools
import cv2
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser(description='Process image folder with via labels as via_export_json.json')
parser.add_argument('path', metavar='PATH', type=str, help='path to folder')

args = parser.parse_args()

gdrive_path = "/content/"


def proc_img(img_dir):
  # load json label file
  json_file = os.path.join(img_dir, "via_export_json.json")
  with open(json_file) as f:
    imgs_anns = json.load(f)

  # process all labels
  filenames = []
  for idx, v in enumerate(imgs_anns.values()):
    filename = os.path.join(img_dir, v["filename"])
    filenames.append(os.path.join(gdrive_path, filename))
    filename_noext = filename.split('.')[0]
    filename_txt = filename_noext + '.txt'

    im_h, im_w = cv2.imread(filename).shape[:2]

    # process bounding boxes
    annos = v["regions"]
    objs = []
    for anno in annos:
      shape = anno["shape_attributes"]
      bb_x = shape["x"]
      bb_y = shape["y"]
      bb_w = shape["width"]
      bb_h = shape["height"]
      categories = anno["region_attributes"]
      if categories["lure"] == "calm":
        class_id = 0
      else:
        class_id = 1

      bb_center_x = bb_x + 1/2 * bb_w
      bb_center_y = bb_y + 1/2 * bb_h

      obj = [
          class_id,
          bb_center_x / im_w,
          bb_center_y / im_h,
          bb_w / im_w,
          bb_h / im_h
      ]
      obj = [str(x) for x in obj]
      line = ' '.join(obj)
      objs.append(line)

    # save labels to txt file
    with open(filename_txt, 'w') as f:
      print(objs)
      f.writelines(f'{s}\n' for s in objs)
  with open('filenames.txt', 'w') as f:
    f.writelines(f'{s}\n' for s in filenames)


#####
#   main
#####

if __name__ == '__main__':
  dir = args.path
  proc_img(dir)
