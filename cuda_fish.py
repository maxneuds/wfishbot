import numpy as np
import cv2
import time
import datetime
import os
import argparse
from copy import copy, deepcopy

import sys
from threading import Thread


from pynput.keyboard import Key
from pynput.mouse import Button


from PIL import Image

from bot_func import capture_screen, get_rect
from bot_func import kb_click, mouse_click_slow, mouse_click_fast
from bot_func import listen
from bot_func import print_status, sleep
from bot_func import save_image, fix_scale_img
from bot_func import draw_detections

#####
# parms
#####

# parser = argparse.ArgumentParser(description='Catch some fish by AI.')
# parser.add_argument('m', metavar='MONID', type=str, help='Display Number')
# args = parser.parse_args()

# yolo size
yolo_size = 416
# target window size and margin
margins = {
    "lr": 0.2,
    "top": 0.15,
    "bot": 0.45
}
# class ids
fish_class = {'calm': 0, 'catch': 1}


#
# dnn settings
#

model_cfg = f"yolo/yolo-fish-{yolo_size}.cfg"
model_weights = f"yolo/yolo-fish-{yolo_size}.weights"
class_names = "yolo/fish.names"
net = cv2.dnn.readNet(model_cfg, model_weights)
# CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# CUDA
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open(class_names, "r") as f:
  classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


##
# globals
##

bot_running = False
trigger_save_image = False
fish_detection_count = 0
t_last_detect = time.time()
t_last_trash = time.time()
t_new_hook = time.time()


##
# functions
##


def on_press(key):
  global bot_running
  global trigger_save_image
  t = Thread(target=robot)
  if key == Key.f9 and not bot_running:
    print('Starting bot...')
    t.start()
  elif key == Key.f12:
    trigger_save_image = True
  elif key == Key.f10:
    # stop loop
    print('Stopping bot...')
    bot_running = False
  elif key == Key.f11:
    # exit script
    bot_running = False
    sys.exit(0)


def feed_dnn(img_cv2, w, h):
  # feed image to dnn
  blob = cv2.dnn.blobFromImage(img_cv2, 1.0/255.0, (yolo_size, yolo_size), (0, 0, 0), True, crop=False)
  net.setInput(blob)
  outs = net.forward(output_layers)
  # add high confidence detections to array
  detections = []
  for out in outs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > 0.8:
        # Object detected
        center_x = int(detection[0] * w)
        center_y = int(detection[1] * h)
        w = int(detection[2] * w)
        h = int(detection[3] * h)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        # add good detection to collection
        detections.append({
            "class_id": class_id,
            "box": [x, y, w, h],
            "confidence": np.around(float(confidence), 4),
            "detection": detection
        })
  return detections


def bot_fish(region, detections):
  # globals
  global fish_detection_count
  global t_last_detect
  global t_last_trash
  global t_new_hook
  # check for successful deteciton
  if len(detections) > 0:
    # get best detection
    best_confidence = 0
    best_idx = 0
    for idx, detection in enumerate(detections):
      if detection["confidence"] > best_confidence:
        best_confidence = detection["confidence"]
        best_idx = idx
    # automate catching
    r_left = region['left']
    r_top = region['top']
    r_w = region['width']
    r_h = region['height']
    if detection["class_id"] == fish_class['catch']:
      # check for fish detection
      if detection["class_id"] == fish_class["catch"]:
        fish_detection_count += 1
      if fish_detection_count > 1:
        # double checked, fish found
        affine_x = int(r_left + detection["detection"][0] * r_w)
        affine_y = int(r_top + detection["detection"][1] * r_h)
        affine_pos = (affine_x, affine_y)
        sleep(371, 553)
        print_status('Fish found! Clicking bobber!')
        mouse_click_slow(affine_pos, Button.right)
        fish_detection_count = 0
        sleep(1293, 1517)
        kb_click(Key.f1)
        t_new_hook = time.time()
    t_last_detect = time.time()
  if time.time() - t_last_trash > 60 * 2:
    kb_click(Key.f2)
    sleep(267, 391)
    kb_click(Key.f3)
    sleep(267, 391)
    print_status('Trash trashed!')
    t_last_trash = time.time()
  if time.time() - t_last_detect > 4:
    print_status('Bobber not found. Throwing new bobber!')
    kb_click(Key.insert)
    sleep(793, 917)
    kb_click(Key.f1)
    t_last_detect = time.time()
    t_new_hook = time.time()


def show_monitor(img_cv2, w, h):
  # def minimum w/h for window
  m_w = 256
  m_h = 128
  if w > m_w:
    m_w = w
  if h > m_h:
    m_h = h
  cv2.imshow('monitor', img_cv2)
  cv2.resizeWindow('monitor', m_w, m_h)
  cv2.waitKey(1)


def robot():
  # set running to true
  global bot_running
  global trigger_save_image
  global t_new_hook
  trigger_save_image = False
  bot_running = True

  last_time = time.time()
  # throw new hook
  kb_click(Key.f1)
  t_new_hook = time.time()
  while(True):
    # break on stop
    if not bot_running:
      cv2.destroyAllWindows()
      break

    rect = get_rect()
    screen, region = capture_screen(rect, margins)
    screen_array = np.asarray(screen)
    img_cv2 = cv2.cvtColor(screen_array, cv2.COLOR_BGRA2BGR)
    img_feed = copy(img_cv2)

    h, w, _ = img_feed.shape

    if time.time() - t_new_hook > 3:
      # feed dnn
      detections = feed_dnn(img_feed, w, h)
      # draw detections
      draw_detections(detections, img_feed, classes)
      # show monitor
      # show_monitor(img_feed, w, h)
      # bot fishing
      bot_fish(region, detections)
    else:
      # show monitor
      # can't continue with monitor output on linux
      # show_monitor(img_feed, w, h)
      pass

    # save image on trigger
    if trigger_save_image:
      save_image(img_feed)
      trigger_save_image = False

    # print fps
    fps = 1/(time.time()-last_time)
    fps = int(np.floor(fps))
    print('{} FPS'.format(fps))
    last_time = time.time()


if __name__ == '__main__':
  listen(on_press)
