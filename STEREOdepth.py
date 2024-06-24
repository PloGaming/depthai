import cv2
import depthai
import numpy
import collections
import time

# Impostazioni generali
CAMERA_FPS = 30
TEXT_OFFSET = 20
TEXT_COLOR = (0, 255, 0)
COLOR_CAMERA_QUEUE_SIZE = 4
QUEUE_BLOCKING = False
CAMERA_INTERLEAVED = False
DISP_STREAM_NAME = "disp"
RGB_STREAM_NAME = "rgb"
DEPTH_STREAM_NAME = "depth"
WINDOW_NAME = "rgb depth"
WINDOW_SIZE = (848, 480)


# Impostazioni depth-related
# Raddoppia il range di disparità per catturare distanze minori (influisce sulle performance)
# OAKD 30cm MIN = False ; 15 cm MIN = True
EXTENDED_DISPARITY = False
# Abilita una maggiore precisione per distanze più lunghe (influisce sulle performance)
SUBPIXEL = False
# Migliora la gestione delle occlusioni (influisce sulle performance)
LRCHECK = True
# Filtra le regioni di bassa fiducia. Va da 0(massimo) a 255(minimo)
THRESHOLD = 255
# Preserva i bordi mentre riduce il rumore.
BILATERAL_SIGMA = 0

# Percentuale rgb/depth nell'immagine
rgbWeight = 0.4
depthWeight = 0.6

# Posizione click mouse
mouseX, mouseY = 0, 0

# Funzione per impostare l'origine del click
def mouseCallback(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked x:", str(x), "; y:", str(y))
        mouseX, mouseY = x,y

# Funzione per regolare la percentuale di rgb e depthmap 
def updateBlendWeights(percent_rgb):
    global depthWeight, rgbWeight
    rgbWeight = float(percent_rgb) / 100.0
    depthWeight = 1.0 - rgbWeight

# Create pipeline
pipeline = depthai.Pipeline()

# Nodi per la pipeline
monoLeft = pipeline.create(depthai.node.MonoCamera) # Mono-camera sinistra
monoRight = pipeline.create(depthai.node.MonoCamera) # Mono-camera destra
depth = pipeline.create(depthai.node.StereoDepth) # Nodo stereodepth
xout = pipeline.create(depthai.node.XLinkOut) # Link-out per inviare risultati al pc


# Imposta il nome del link-out
xout.setStreamName(DISP_STREAM_NAME)


# Imposta la risoluzione delle videocamere
monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)

# Imposta le videocamere
monoLeft.setBoardSocket(depthai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(depthai.CameraBoardSocket.CAM_C)

# Imposta gli FPS
monoLeft.setFps(CAMERA_FPS)
monoRight.setFps(CAMERA_FPS)

# E' un filtro che "corregge gli errori" dell'immagine originale eliminando gli artefatti con il valore mediano dei pixel vicini
depth.initialConfig.setMedianFilter(depthai.MedianFilter.MEDIAN_OFF)
depth.initialConfig.PostProcessing.SpeckleFilter.enable = True
depth.initialConfig.setConfidenceThreshold(THRESHOLD)
depth.initialConfig.setBilateralFilterSigma(BILATERAL_SIGMA)

# Queste erano le opzioni definite prima che vengono impostate
depth.setLeftRightCheck(LRCHECK)
depth.setExtendedDisparity(EXTENDED_DISPARITY)
depth.setSubpixel(SUBPIXEL)

depth.setDepthAlign(depthai.CameraBoardSocket.CAM_A)
depth.setOutputSize(1920, 1080)

RGBCamera = pipeline.create(depthai.node.ColorCamera)
RGBCamera.setBoardSocket(depthai.CameraBoardSocket.CAM_A)
RGBCamera.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
RGBCamera.setInterleaved(CAMERA_INTERLEAVED)
RGBCamera.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
RGBCamera.setFps(CAMERA_FPS)
rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName(RGB_STREAM_NAME)

# Collegamenti
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)
RGBCamera.video.link(rgbOut.input)

LinkOutDepth = pipeline.createXLinkOut()
LinkOutDepth.setStreamName(DEPTH_STREAM_NAME)
depth.depth.link(LinkOutDepth.input)

# "Passa" al dispositivo la pipeline appena creata
with depthai.Device(pipeline) as OAKD:
    frameRgb = None
    frameDisp = None
    frameDepth = None

    # Crea la finestra di nome "oakd"
    cv2.namedWindow(WINDOW_NAME)
    #cv2.resizeWindow(WINDOW_NAME, 848, 480)

    # Crea la trackbar per regolare depth - rgb
    cv2.createTrackbar('RGB Depth', WINDOW_NAME, int(rgbWeight * 100), 100, updateBlendWeights)
    
    # Imposta un callback per l'immagine
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

    print('Usb speed: ', OAKD.getUsbSpeed().name)
    print('Device name: ', OAKD.getDeviceName())
    print('Device information: ', OAKD.getDeviceInfo())

    # Prende i frame dalla coda di output del Link-out
    queueStereo = OAKD.getOutputQueue(name=DISP_STREAM_NAME, 
                            maxSize=COLOR_CAMERA_QUEUE_SIZE, 
                            blocking=QUEUE_BLOCKING)

    queueRGB = OAKD.getOutputQueue(
           name=RGB_STREAM_NAME,
           maxSize=COLOR_CAMERA_QUEUE_SIZE,
           blocking=QUEUE_BLOCKING)

    queueDepth = OAKD.getOutputQueue(
        name=DEPTH_STREAM_NAME,
        maxSize=COLOR_CAMERA_QUEUE_SIZE,
        blocking=QUEUE_BLOCKING)

    while True:
        lastFrame = {}
        lastFrame["rgb"] = queueRGB.tryGet()
        lastFrame["disp"] = queueStereo.tryGet()
        lastFrame["depth"] = queueDepth.tryGet()        

        if lastFrame["rgb"] is not None:
            frameRgb = lastFrame["rgb"].getCvFrame()

        if lastFrame["disp"] is not None:
            frameDisp = lastFrame["disp"].getFrame()
            maxDisparity = depth.initialConfig.getMaxDisparity()
            frameDisp = (frameDisp * 255. / maxDisparity).astype(numpy.uint8)
            frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)

        if lastFrame["depth"] is not None:
            frameDepth = lastFrame["depth"].getFrame()
            depth_value_mm = frameDepth[mouseY, mouseX]

        if frameRgb is not None and frameDisp is not None:
            if len(frameDisp.shape) < 3:
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
            
            #Resize immagine per averla più piccola
            #blended = cv2.resize(blended, (WINDOW_SIZE[0], WINDOW_SIZE[1]))
            
            cv2.putText(blended, f"Depth ({mouseX},{mouseY}): {depth_value_mm} mm", 
                            (0, 3 * TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            
            cv2.imshow(WINDOW_NAME, blended)
            
            frameRgb = None
            frameDisp = None
            frameDepth = None

        # Premere 'q' per uscire
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            exit()
            