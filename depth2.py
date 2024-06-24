import depthai as dai
import numpy as np
import cv2

# Impostazioni generali
CAMERA_FPS = 120
TEXT_OFFSET = 30
TEXT_COLOR = (255, 255, 255)
COLOR_CAMERA_QUEUE_SIZE = 4
QUEUE_BLOCKING = False
STREAM_NAME = "depth"

# Impostazioni depth-related
# Raddoppia il range di disparità per catturare distanze minori (influisce sulle performance)
# OAKD 30cm MIN = False ; 15 cm MIN = True
EXTENDED_DISPARITY = False
# Abilita una maggiore precisione per distanze più lunghe (influisce sulle performance)
SUBPIXEL = False
# Migliora la gestione delle occlusioni (influisce sulle performance)
LRCHECK = False
# Filtra le regioni di bassa fiducia. Va da 0(massimo) a 255(minimo)
THRESHOLD = 200  #prima era 170
# Preserva i bordi mentre riduce il rumore.
BILATERAL_SIGMA = 50000


# Global variables
mouseX, mouseY = 0, 0
TEXT_OFFSET = 20
TEXT_COLOR = (255, 255, 255)  # White color for text

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked x:", str(x), "; y:", str(y))
        mouseX, mouseY = x, y

# Create a pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName(STREAM_NAME)

# Imposta la risoluzione delle videocamere
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Imposta gli FPS
monoLeft.setFps(CAMERA_FPS)
monoRight.setFps(CAMERA_FPS)

# Configurations for stereo depth
stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
stereo.initialConfig.PostProcessing.SpeckleFilter.enable = True
stereo.initialConfig.setConfidenceThreshold(THRESHOLD)
stereo.initialConfig.setBilateralFilterSigma(BILATERAL_SIGMA)
stereo.setLeftRightCheck(LRCHECK)
stereo.setExtendedDisparity(EXTENDED_DISPARITY)
stereo.setSubpixel(SUBPIXEL)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the depth frames from the outputs defined above
    qDepth = device.getOutputQueue(name=STREAM_NAME, maxSize=4, blocking=False)

    # Set up the OpenCV window and mouse callback
    cv2.namedWindow(STREAM_NAME)
    cv2.setMouseCallback(STREAM_NAME, mouse_callback)

    while True:
        inDepth = qDepth.get()  # Get depth frame
        depthFrame = inDepth.getFrame()  # Retrieve depth frame as numpy array

        # Normalize and colorize the depth frame for display
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        # Access the depth value at the current mouse position
        if 0 <= mouseX < depthFrame.shape[1] and 0 <= mouseY < depthFrame.shape[0]:
            depth_value_mm = depthFrame[mouseY, mouseX]
            cv2.putText(depthFrameColor, f"Depth ({mouseX},{mouseY}): {depth_value_mm} mm", 
                        (0, 3 * TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Display the depth map
        cv2.imshow(STREAM_NAME, depthFrameColor)

        # Wait for a key press
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()