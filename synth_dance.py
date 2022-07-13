import cv2 as cv
import numpy as np 
import pygame.midi as midi
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
  cam_width = 1280#1920
  global cam_height
  cam_height = 720#1080

  global player_function
  player_function = lambda x: 0 if x < 2 else 31 if x < 4 else 64 if x < 6 else 96 if x < 8 else 127

  # Open the webcam
  
  global cam
  cam = None
  try:
    cam = cv.VideoCapture(4)
    print(cv.CAP_PROP_FRAME_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_WIDTH,cam_width)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT,cam_height)
    img = cam.read()
    print(img[1].shape[0], img[1].shape[1])
  except:
    print("no cam available")
  pb  = 'frozen_inference_graph.pb'
  pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
  
  # Read the neural network
  global cvNet
  cvNet = cv.dnn.readNetFromTensorflow(pb,pbt)

  fm_port = 2
  beat_port = 4

  midi.init()
  global fm_output
  fm_output = None
  try:
    fm_output = midi.Output(fm_port)
  except midi.MidiException:
    print("Midi port {fm_port} unavailable for FM".format(fm_port=fm_port))

  global beat_output
  beat_output = None
  try:
    beat_output = midi.Output(beat_port)
  except midi.MidiException:
    print("Midi port {beat_port} unavailable for beat".format(beat_port=fm_port))


  global bounce_balls
  bounce_balls = []
  for _ in range(5):
    bounce_balls.append({
      "position":(200, random.randint(-200, 0)),
      "radius":random.randint(5, 30),
      "color":(random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)),
      "velocity":(random.choice([-7.0, -5.0,-3.0, 3.0,  5.0, 7.0]), 0.0)})



def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports
#list_ports()



def detect_objects(img, counter, cached_detections):
  if counter < 5:
    counter += 1
    return img, counter, cached_detections
  counter = 0
  cached_detections = []
  cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
 
  # Run object detection
  cvOut = cvNet.forward()
 
  # Go through each object detected and label it
  for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
 
      idx = int(detection[1])
      if classes[idx] in ['person', 'giraffe', 'broccoli', 'elephant', 'dog', 'stop sign']:        
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
          'idx':idx,
          'score':score})

        y = top - 15 if top - 15 > 15 else top + 15
        vals = None

        centerX = left + (right - left) / 2
        centerY = top + (bottom - top) / 2
        if  centerX < cam_width / 2:
          if centerY < cam_height / 2:
            #LFO
            vals = sendXYControlChange(fm_output, centerX, centerY, 46, 47)
          else:
            #CARRIER
            vals = sendXYControlChange(fm_output, centerX, centerY - cam_height / 2, 44, 45)
        else:
          if top + (bottom - top) / 2 < cam_height / 2:
            #VELOCITY
            vals = sendXYControlChange(fm_output, centerX - cam_width / 2, centerY, 41, 49)
          else:
            #MODULATOR
            vals = sendXYControlChange(fm_output, centerX - cam_width / 2, centerY - cam_height / 2, 42, 43)

        if vals != None:
            cv.putText(img, vals, (int(centerX), int(centerY)), cv.FONT_HERSHEY_SIMPLEX, 1, (220, 100, 100), thickness = 2)

  sendBeatControlChanges(len(cached_detections))
  #sendYControlChange(41, highest_detection)
  #sendRangedControlChange(49, len(cached_detections), player_function)
  return img, counter, cached_detections


def sendBeatControlChanges(count):
  kick_vol = 127
  snare_vol = 127
  lo_tom_vol = 127
  hi_tom_vol = 127
  cl_hat_vol = 127
  op_hat_vol = 127
  clap_vol = 127
  if count < 5:
    lo_tom_vol = 0
    hi_tom_vol = 0
  if count < 4:
    snare_vol = 0
    clap_vol = 0
  if count < 3:
    cl_hat_vol = 63
    op_hat_vol = 63
  if count < 2:
    cl_hat_vol = 0
    op_hat_vol = 0
    kick_vol = 63
  if count < 1:
    kick_vol =0
  sendYControlChange(beat_output, 40, kick_vol)
  sendYControlChange(beat_output, 41, snare_vol)
  sendYControlChange(beat_output, 42, lo_tom_vol)
  sendYControlChange(beat_output, 43, hi_tom_vol)
  sendYControlChange(beat_output, 44, cl_hat_vol)
  sendYControlChange(beat_output, 45, op_hat_vol)
  sendYControlChange(beat_output, 46, clap_vol)

def sendXYControlChange(output, centerX, centerY, first, second):
  x = int((centerX / (cam_width / 2)) * 127)
  y = int((centerY / (cam_height / 2)) * 127)

  vals = str("x: "+ str(x) +  " y: " + str(y))
  if (output != None):
    output.write_short(0xb0, first, min(127, max(0, x)))
    output.write_short(0xb0, second, min(127, max(0, y)))
    #print(vals)
  return vals

def sendYControlChange(output, type, value):
  #y = 127 - int((value / cam_height) * 127)
  if output != None:
    output.write_short(0xb0, type, min(127, max(0, value)))

def sendRangedControlChange(output, type, value, range_function):
  rang = range_function(value)
  if output != None:
    output.write_short(0xb0, type, rang)


def draw_detections(img, cached_detections):
  for detection in cached_detections:
    left = detection['left']
    right = detection['right']
    top = detection['top']
    bottom = detection['bottom']
    idx = detection['idx']
    score = detection['score']
    cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    cv.circle(img, (int(left + (right - left) / 2), int(top + cam_height / 10)), int(cam_height / 20), (23, 230, 210), thickness=-1)

    # draw the prediction on the frame
    label = "{}: {:.2f}%".format(classes[idx],score * 100)
    y = top - 15 if top - 15 > 15 else top + 15
    cv.putText(img, label, (int(left + (right - left) / 2), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)


def draw_bounce_balls(img, cached_detections):
  for ball in bounce_balls:
    ball["position"] = (ball["position"][0] + ball["velocity"][0], ball["position"][1] + ball["velocity"][1])
    posX = ball["position"][0]
    posY = ball["position"][1]
    velX = ball["velocity"][0]
    velY = ball["velocity"][1]
    if posX > cam_width or posX < 0:
      velX = velX * - 1
    if posY > cam_height:
      velY = velY * -1
    elif posY < - 200:
      velY = velY * - 0.1
    if posX < -200 or posX > cam_width * 2:
      posX = 200
    if posY < -400 or posY > cam_height * 2:
      posY = -100
    cv.circle(img, (int(posX), int(posY)), ball["radius"], ball["color"], thickness=4)

    for detection in cached_detections:
      if posX < detection["right"] and posX > detection["left"] and posY < detection["bottom"] and posY > detection["top"]:
        if posX - velX - 1 > detection["right"] or posX + velX + 1 < detection["left"]:
          velX = velX * - 1.5
        if posY + velY + 1 > detection["bottom"] or posY - velY - 1 < detection["top"]:
          velY = velY * - 1.5

    ball["velocity"] = (velX, velY + 0.2)



def start():
  counter = 0
  cached_detections = []
  count = 0
  while True:
    #count += 1
    ret_val, img = cam.read()
    img = cv.flip(img, 1)
    img, counter, cached_detections = detect_objects(img, counter, cached_detections)

    #img_old = img
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #img = cv.GaussianBlur(img, (3,3), 0)
    #img = cv.Canny(image=img, threshold1=50, threshold2=20)
    #ret, logo_mask = cv.threshold(img[:,0], 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    #img_old[np.where(logo_mask == 255)] = img[np.where(logo_mask == 255)]

    draw_detections(img, cached_detections)
    draw_bounce_balls(img, cached_detections)
    # Display the frame
    cv.line(img, (int(cam_width / 2), 0), (int(cam_width / 2), cam_height), (220, 220, 220), 5)
    cv.line(img, (0, int(cam_height / 2)), (cam_width, int(cam_height / 2)), (220, 220, 220), 5)
    cv.putText(img, "LFO", (int(10), int(cam_height / 2 - 10)),cv.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
    cv.putText(img, "VELOCITY", (int(cam_width / 2 + 10), int(cam_height / 2 - 10)),cv.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
    cv.putText(img, "CARRIER", (int(10), int(cam_height - 10)),cv.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
    cv.putText(img, "MODULATOR", (int(cam_width / 2 + 10), int(cam_height - 10)),cv.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)

    img = rescale_frame(img, percent=150)
    cv.imshow('DANCE LITTLE MONKEY', img)

    if cv.waitKey(1) == 27:
      break


  cam.release()
  cv.destroyAllWindows()