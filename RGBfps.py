import depthai
import cv2
import numpy

COLOR_CAMERA_PREVIEW_SIZE = 300, 300
CAMERA_INTERLEAVED = False
CAMERA_FPS = 60
TEXT_OFFSET = 16
COLOR_CAMERA_QUEUE_SIZE = 1
QUEUE_BLOCKING = False
TEXT_COLOR = (255, 255, 255)

# Crea pipeline
pipeline = depthai.Pipeline()

# Crea camera RGB
RGBCamera = pipeline.create(depthai.node.ColorCamera)
RGBCamera.setPreviewSize(COLOR_CAMERA_PREVIEW_SIZE)
RGBCamera.setBoardSocket(depthai.CameraBoardSocket.CAM_A)
RGBCamera.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
RGBCamera.setInterleaved(CAMERA_INTERLEAVED)
RGBCamera.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
RGBCamera.setFps(CAMERA_FPS)
LinkOutCamera = pipeline.createXLinkOut()
LinkOutCamera.setStreamName("rgb")
RGBCamera.preview.link(LinkOutCamera.input)

# Crea finestre di nome "rgb"
cv2.namedWindow("rgb")

# Assegna la pipeline appena creata al dispositivo
with depthai.Device(pipeline) as OAKD:
    
    print('Usb speed: ', OAKD.getUsbSpeed().name)
    print('Device name: ', OAKD.getDeviceName())
    print('Device information: ', OAKD.getDeviceInfo())
    

    queueRGB = OAKD.getOutputQueue(
           name='rgb',
           maxSize=COLOR_CAMERA_QUEUE_SIZE,
           blocking=QUEUE_BLOCKING,
       )

    startTime = depthai.Clock.now().total_seconds()
    counterFrames = 0
    fps = 0

    while(True):
        inputRGB = queueRGB.get().getCvFrame()
        
        counterFrames += 1
        current_time = depthai.Clock.now().total_seconds()
        if (current_time - startTime) >= 1 :
            fps = counterFrames / (current_time - startTime)
            counterFrames = 0
            startTime = current_time

        cv2.putText(inputRGB, "fps: " + str(int(fps*1000) / 1000), 
                    (0, 0+TEXT_OFFSET), cv2.FONT_ITALIC, 0.5 ,
                    TEXT_COLOR, 1, cv2.LINE_AA)

        cv2.imshow('rgb', inputRGB)
        
        if cv2.waitKey(1) == ord(chr(27).encode()): 
            cv2.destroyAllWindows()
            exit()
            break
        