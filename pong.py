from sre_parse import State
import cv2 as cv
import numpy as np
import pygame
import random

def init():

  global classes
  classes = ["background", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
    "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
    "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
  
  # Colors we will use for the object labels
  global colors
  colors = np.random.uniform(0, 255, size=(len(classes), 3))

  global cam_width
  cam_width = 1920
  global cam_height
  cam_height = 1080

  
  global cam
  cam = None
  try:
    cam = cv.VideoCapture(4)
    print(cv.CAP_PROP_FRAME_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_WIDTH,1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
    img = cam.read()
    print(img[1].shape[0], img[1].shape[1])
  except:
    print("no cam available")
  pb  = 'frozen_inference_graph.pb'
  pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
  
  # Read the neural network
  global cvNet
  cvNet = cv.dnn.readNetFromTensorflow(pb,pbt)

  global pedals
  pedals =[
    pygame.Vector2(50, cam_height / 2),
    pygame.Vector2(250, cam_height / 2)]

  global balls
  balls = [{
    'position':pygame.Vector2(cam_width / 2, cam_height / 2),
    'velocity':pygame.Vector2(5, 5)}]

  global particles
  particles = []

  global state
  state = {
    'left':0,
    'right':0
  }


def detect_objects(img, counter, cached_detections):
  if counter < 9:
    counter += 1
    return img, counter, cached_detections
  counter = 0
  cached_detections = []
  blob = cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
  cvNet.setInput(blob)
 
  # Run object detection
  cvOut = cvNet.forward()
  highest_detection = cam_height
 
  # Go through each object detected and label it
  for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
 
      idx = int(detection[1])
      if classes[idx]:# in ['person', 'dog', 'cell phone']:        
        left = detection[3] * cam_width
        top = detection[4] * cam_height
        right = detection[5] * cam_width
        bottom = detection[6] * cam_height
        cached_detections.append(
          {'classIndex':classes[idx],
          'left':left,
          'top':top,
          'right':right,
          'bottom':bottom,
          'middle':pygame.Vector2(left + (right - left) / 2, top + (bottom - top) / 2),
          'idx':idx,
          'score':score})

  return img, counter, cached_detections


def draw_detections(img, cached_detections):
  for detection in cached_detections:
    left = detection['left']
    right = detection['right']
    top = detection['top']
    bottom = detection['bottom']
    idx = detection['idx']
    score = detection['score']
    cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

    # draw the prediction on the frame
    label = "{}: {:.2f}%".format(classes[idx],score * 100)
    y = top - 15 if top - 15 > 15 else top + 15
    cv.putText(img, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)


def draw_pedals(img, cached_detections):
  pedal1drawn = False
  pedal2drawn = False
  for detection in cached_detections:
    if not pedal1drawn and detection['classIndex'] == 'cell phone' and detection['right'] < cam_width / 2:
      pedal1drawn = True
      pedals[0].x = detection['middle'].x
      pedals[0].y = detection['middle'].y
      
    elif not pedal2drawn and detection['classIndex'] == 'cell phone' and detection['left'] > cam_width / 2:
      pedal2drawn = True
      pedals[1].x = detection['middle'].x
      pedals[1].y = detection['middle'].y
  cv.rectangle(img,
    (int(pedals[1].x - 10), int(pedals[1].y - 30)),
    (int(pedals[1].x + 10), int(pedals[1].y + 30)),
    (23, 230, 10), thickness=-1)
  cv.rectangle(img,
    (int(pedals[0].x - 10), int(pedals[0].y - 30)),
    (int(pedals[0].x + 10), int(pedals[0].y + 30)),
    (23, 30, 210), thickness=-1)


def draw_particles(img):
  remove = []
  for particle in particles:
    particle['life'] = particle['life'] - 1
    if particle['life'] == 0:
      remove.append(particle)
      continue
    particle['position'] = particle['position'].__add__(particle['velocity'])
    cv.circle(img,
      (int(particle['position'].x), int(particle['position'].y)),
      int(particle['life'] / 5), (200, 30, 220), thickness=-1)
  for rem in remove:
    particles.remove(rem)


def spawn_particles(position):
  for _ in range(20):
    particles.append({
      'position':pygame.Vector2(position.x, position.y),
      'velocity':pygame.Vector2(
          int(random.randint(-100, 100)),
          int(random.randint(-100, 100)))
            .normalize()
            .__mul__(3.0),
      'life':30
    })


def draw_balls(img, cached_detections):
  for ball in balls:
    ball['position'] = ball['position'].__add__(ball['velocity'])
    if ball['position'].y < 10 or ball['position'].y > cam_height - 10:
      ball['velocity'].y = ball['velocity'].y * -1
    elif ball['position'].x < -10:
      ball['velocity'].x = ball['velocity'].x * -1
      state['right'] = state['right'] + 1 
      spawn_particles(ball['position'])
    elif ball['position'].x > cam_width + 10:
      ball['velocity'].x = ball['velocity'].x * -1 
      state['left'] = state['left'] + 1
      spawn_particles(ball['position'])
    elif (ball['position'].x < pedals[1].x
      and ball['position'].x > pedals[1].x - 20
      and ball['position'].y > pedals[1].y - 40
      and ball['position'].y < pedals[1].y + 40
      and ball['velocity'].x > 0):

      distance = (pedals[1].y - ball['position'].y) * 1 if ball['velocity'].y > 0 else -1
      print(distance)
      ball['velocity'] = pygame.Vector2(-40, -distance).normalize().__mul__(5.0)
      #ball['velocity'].x = ball['velocity'].x * -1   
      spawn_particles(ball['position'])
    elif (ball['position'].x > pedals[0].x
      and ball['position'].x < pedals[0].x + 20
      and ball['position'].y > pedals[0].y - 40
      and ball['position'].y < pedals[0].y + 40
      and ball['velocity'].x < 0):
      distance = (pedals[1].y - ball['position'].y) * 1 if ball['velocity'].y > 0 else -1
      print(distance)
      ball['velocity'] = pygame.Vector2(40, -distance).normalize().__mul__(5.0)
      #ball['velocity'].x = ball['velocity'].x * -1
      spawn_particles(ball['position'])

    cv.circle(img,
      (int(ball['position'].x), int(ball['position'].y)),
      10, (200, 30, 30), thickness=-1)


def draw_state(img):
  cv.line(img, (int(cam_width / 2), 0), (int(cam_width / 2), cam_height), (220, 220, 220), thickness=4)
  cv.putText(img, str(state['left']), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (220, 220, 220), 2)
  cv.putText(img, str(state['right']), (cam_width - 40, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (220, 220, 220), 2)


def start():
  counter = 0
  cached_detections = []
  while True:
    ret_val, img = cam.read()
    img = cv.flip(img, 1)
    img, counter, cached_detections = detect_objects(img, counter, cached_detections)
    draw_detections(img, cached_detections)
    draw_pedals(img, cached_detections)
    draw_balls(img, cached_detections)
    draw_particles(img)
    draw_state(img)
    # Display the frame

    #img = rescale_frame(img, percent=250)
    cv.imshow('PONG PONG BABY', img)

    if cv.waitKey(1) == 27:
      break


  cam.release()
  cv.destroyAllWindows()