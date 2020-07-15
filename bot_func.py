import os
import time
import cv2
import datetime
import numpy as np
from mss import mss
from random import randint
from PIL import Image
from pynput.mouse import Controller as mouse_controller
from pynput.keyboard import Key, KeyCode, Controller as kb_controller, Listener as kb_listener

if os.name == 'nt':
  import win32gui
else:
  from ewmh import EWMH
  ewmh = EWMH()


#####
#
# Screencapture
#
#####


def linux_get_rect():
  def frame(win):
    frame = win
    while frame.query_tree().parent != ewmh.root:
      frame = frame.query_tree().parent
    return frame
  try:
    win = ewmh.getActiveWindow()
    frame = frame(win)
    geo = frame.get_geometry()
    left = geo.x
    top = geo.y
    w = geo.width
    h = geo.height
    if w < 256:
      w = 256
    if h < 128:
      h = 128
    out_rect = (left, top, w, h)
    return out_rect
  except:
    time.sleep(1)
    return (400, 400, 400, 400)


def windows_get_rect():
  try:
    hwnd = win32gui.GetForegroundWindow()
    rect = win32gui.GetWindowRect(hwnd)
    # win32gui rect = (left, top, right, bot)
    # transform to (left, top, width, height)
    left = rect[0]
    top = rect[1]
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    if w < 256:
      w = 256
    if h < 128:
      h = 128
    out_rect = (left, top, w, h)
    return out_rect
  except:
    time.sleep(1)
    return (400, 400, 400, 400)


def get_rect():
  if os.name == 'nt':
    return windows_get_rect()
  else:
    return linux_get_rect()


#####
#
# Mouse & Keyboard
#
#####


mouse = mouse_controller()
kb = kb_controller()


def mouse_click_slow(pos, button):
  mouse.position = pos
  sleep(1377, 1513)
  mouse.click(button)
  sleep(51, 79)


def mouse_click_fast(pos, button):
  mouse.position = pos
  sleep(3, 17)
  mouse.click(button)
  sleep(11, 23)


def kb_click(key):
  kb.press(key)
  sleep(1, 11)
  kb.release(key)


def listen(on_press):
  try:
    with kb_listener(on_press=on_press) as l:
      l.join()
  except KeyboardInterrupt:
    pass


#####
#
# General
#
#####


def sleep(min, max):
  time.sleep(randint(min, max)/1000)


def print_status(text):
  t_now = datetime.datetime.now()
  t_now_str = t_now.strftime('%H:%M:%S')
  print(f'{t_now_str} > {text}')


#####
#
# Visuals
#
#####

def capture_screen(rect, margins):
  margin_lr = margins["lr"]
  margin_top = margins["top"]
  margin_bot = margins["bot"]

  left = rect[0]
  top = rect[1]
  w = rect[2]
  h = rect[3]

  ratio = w / h

  # capture region
  c_top = int(top + margin_top * h)
  c_left = int(left + margin_lr * ratio * w)
  c_w = int(w * (1 - 2 * margin_lr * ratio))
  c_h = int(h * (1 - margin_top - margin_bot))
  region = {
      "top": c_top,
      "left": c_left,
      "width": c_w,
      "height": c_h
  }
  with mss() as sct:
    screen = sct.grab(region)
  return screen, region


def scale_img(img_cv2, img_scale):
  h, w, _ = img_cv2.shape
  x_new, y_new = w*img_scale, h*img_scale
  img_new = cv2.resize(img_cv2, (int(x_new), int(y_new)))
  return img_new


def fix_scale_img(img_cv2, max_size):
  h, w, _ = img_cv2.shape
  if w > max_size:
    img_scale = max_size / w
  else:
    img_scale = 1
  x_new, y_new = w*img_scale, h*img_scale
  img_new = cv2.resize(img_cv2, (int(x_new), int(y_new)))
  return img_new


# draw detection to image
def draw_detections(detections, cv_img, classes):
  font = cv2.FONT_HERSHEY_SIMPLEX
  colors = np.array([[0.0, 0.0, 0.0], ]*len(classes))
  for i in range(len(detections)):
    x, y, w, h = detections[i]["box"]
    class_str = str(classes[detections[i]["class_id"]])
    confidence = str(round(detections[i]["confidence"], 2))
    label = class_str + " " + confidence
    label_size = cv2.getTextSize(label, font, 0.5, 1)
    label_w = label_size[0][0]
    label_background_w = max(label_size[0][0] + 10, w + 1)
    color = colors[detections[i]["class_id"]]
    cv2.rectangle(cv_img, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(cv_img, (x - 1, y), (x + label_background_w, y - 30), color, -1)
    cv2.putText(cv_img, label, (x + 5, y - 10), font, 0.5, (33, 124, 255), 1)


def save_image(img_cv2):
  pil_im = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
  pil_im = Image.fromarray(pil_im)
  pil_im.save(f'new_data/pool/wow_train_{time.time()}.png')
  print('Train image saved.')
