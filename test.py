#!/usr/bin/env python3

import cv2
import numpy
import depthai
from datetime import timedelta
import time

MS_THRESHOLD = 11

EXTENDED_DISPARITY = False
CAMERA_QUEUE_SIZE = 4
QUEUE_BLOCKING = True
SUBPIXEL = False
LRCHECK = True
THRESHOLD = 255
BILATERAL_SIGMA = 0
WINDOW_NAME = "rgb/depth"
STREAM_NAME = "sync"

rgbWeight = 0.4
depthWeight = 0.6
maxDisparity = 0
messages = dict()

class FPSCounter:
    def __init__(self):
        self.frameTimes = []

    def tick(self):
        now = time.time()
        self.frameTimes.append(now)
        self.frameTimes = self.frameTimes[-100:]

    def getFps(self):
        if len(self.frameTimes) <= 1:
            return 0
        return (len(self.frameTimes) - 1) / (self.frameTimes[-1] - self.frameTimes[0])

def add_msg(msg, name, timestamp = None):
    if timestamp is None:
        timestamp = msg.getTimestamp()

    if not name in messages:
        messages[name] = []

    messages[name].append((timestamp, msg))

    synced = {}
    for name, arr in messages.items():
        # Go through all stored messages and calculate the time difference to the target msg.
        # Then sort these messages to find a msg that's closest to the target time, and check
        # whether it's below 17ms which is considered in-sync.
        diffs = []
        for i, (msg_ts, msg) in enumerate(arr):
            diffs.append(abs(msg_ts - timestamp))
        if len(diffs) == 0: break
        diffsSorted = diffs.copy()
        diffsSorted.sort()
        dif = diffsSorted[0]

        if dif < timedelta(milliseconds=MS_THRESHOLD):
            # print(f'Found synced {name} with timestamp {msg_ts}, target timestamp {timestamp}, diff {dif}, location {diffs.index(dif)}')
            # print(diffs)
            synced[name] = diffs.index(dif)


    if len(synced) == 3: # We have 3 synced messages (IMU packet + disp + rgb)
        # print('--------\Synced messages! Target timestamp', timestamp, )
        # Remove older messages
        for name, i in synced.items():
            messages[name] = messages[name][i:]
        ret = {}
        for name, arr in messages.items():
            ret[name] = arr.pop(0)
            # print(f'{name} msg timestamp: {ret[name][0]}, diff {abs(timestamp - ret[name][0]).microseconds / 1000}ms')
        return ret
    return False

def updateBlendWeights(percent_rgb):
    global depthWeight, rgbWeight
    rgbWeight = float(percent_rgb) / 100.0
    depthWeight = 1.0 - rgbWeight

def create_pipeline(device):
    pipeline = depthai.Pipeline()

    camRgb = pipeline.create(depthai.node.ColorCamera)
    camRgb.setBoardSocket(depthai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(30)
    camRgb.setIspScale(2, 3)
    
    # Calibrazione posizione lenti
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(depthai.CameraBoardSocket.CAM_A)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)

    left = pipeline.create(depthai.node.MonoCamera)
    left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
    left.setBoardSocket(depthai.CameraBoardSocket.CAM_B)
    left.setFps(45)

    right = pipeline.create(depthai.node.MonoCamera)
    right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
    right.setBoardSocket(depthai.CameraBoardSocket.CAM_C)
    right.setFps(45)

    stereo = pipeline.create(depthai.node.StereoDepth)
    
    stereo.initialConfig.PostProcessing.SpeckleFilter.enable = True
    stereo.initialConfig.setMedianFilter(depthai.MedianFilter.MEDIAN_OFF)
    stereo.initialConfig.setConfidenceThreshold(THRESHOLD)
    stereo.initialConfig.setBilateralFilterSigma(BILATERAL_SIGMA)
    stereo.setLeftRightCheck(LRCHECK)
    stereo.setExtendedDisparity(EXTENDED_DISPARITY)
    stereo.setSubpixel(SUBPIXEL)
    stereo.setDepthAlign(depthai.CameraBoardSocket.CAM_A)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    sync = pipeline.create(depthai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=MS_THRESHOLD))

    camRgb.isp.link(sync.inputs["rgb"])
    stereo.depth.link(sync.inputs["depth"])

    imu = pipeline.create(depthai.node.IMU)
    imu.enableIMUSensor(depthai.IMUSensor.ACCELEROMETER_RAW, 360)
    imu.setBatchReportThreshold(10)
    imu.setMaxBatchReports(10)
    imu.out.link(sync.inputs["imu"])

    xout = pipeline.createXLinkOut()
    xout.setStreamName(STREAM_NAME)
    sync.out.link(xout.input) 

    return pipeline

def td2ms(td) -> int:
    # Convert timedelta to milliseconds
    return int(td / timedelta(milliseconds=1))

def new_msg(msg, name, timestamp=None):
    global maxDisparity
    synced = add_msg(msg, name, timestamp)

    if not synced: return

    fps.tick()
    print('FPS', fps.getFps())
    rgb_ts, rgb = synced['rgb']
    stereo_ts, disp = synced['disp']
    imuTs, imu = synced['imu']

    frameRgb = rgb.getCvFrame()
    frameDisp = disp.getFrame()
    
    frameDisp = (frameDisp * 255. / maxDisparity).astype(numpy.uint8)

    frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_TURBO)
    frameDisp = numpy.ascontiguousarray(frameDisp)

    # Need to have both frames in BGR format before blending
    if len(frameDisp.shape) < 3:
        frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
    cv2.imshow(WINDOW_NAME, blended)


# Connect to device and start pipeline
device = depthai.Device()
device.startPipeline(create_pipeline(device))

# Configure windows; trackbar adjusts blending ratio of rgb/depth
cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar('RGB Weight %', WINDOW_NAME, int(rgbWeight*100), 100, updateBlendWeights)

print('Connection type: ', device.getUsbSpeed().name)
print('Device name: ', device.getDeviceName())
print('Device information: ', device.getDeviceInfo())

queue = device.getOutputQueue(
    STREAM_NAME, 
    CAMERA_QUEUE_SIZE, 
    QUEUE_BLOCKING)

maxDisparity = stereo.initialConfig.getMaxDisparity()

fps = FPSCounter()
while True:
    msgGrp = queue.tryGet()

    if msgGrp is not None:
        fps.tick()
        for name, msg in msgGrp:
            print("fps: " + str(fps.getFps()))
            if name == "depth":
                frameDisp = (msg.getFrame() * 255. / maxDisparity).astype(numpy.uint8)
                frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)
                cv2.imshow("depth", frameDisp)
            elif name == "rgb":
                cv2.imshow("rgb", msg.getCvFrame())
    else:
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            exit(0)
            break
