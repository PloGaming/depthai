import cv2
import depthai
import numpy
import collections
import time

# Impostazioni generali
CAMERA_FPS = 30
TEXT_OFFSET = 20
TEXT_COLOR = (0, 255, 0)
CAMERA_QUEUE_SIZE = 4
QUEUE_BLOCKING = False
CAMERA_INTERLEAVED = False
DISP_STREAM_NAME = "disp"
DEPTH_STREAM_NAME = "depth"
WINDOW_NAME = "depth"
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

# Posizione click mouse
mouseX, mouseY = 0, 0

# Funzione per impostare l'origine del click
def mouseCallback(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked x:", str(x), "; y:", str(y))
        mouseX, mouseY = x,y

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

depth.setOutputSize(1920, 1080)

# Collegamenti
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)

LinkOutDepth = pipeline.createXLinkOut()
LinkOutDepth.setStreamName(DEPTH_STREAM_NAME)
depth.depth.link(LinkOutDepth.input)

# "Passa" al dispositivo la pipeline appena creata
with depthai.Device(pipeline) as OAKD:
    frameDisp = None
    frameDepth = None

    # Crea la finestra di nome "oakd"
    cv2.namedWindow(WINDOW_NAME)
    
    # Imposta un callback per l'immagine
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

    print('Usb speed: ', OAKD.getUsbSpeed().name)
    print('Device name: ', OAKD.getDeviceName())
    print('Device information: ', OAKD.getDeviceInfo())

    # Prende i frame dalla coda di output del Link-out
    queueStereo = OAKD.getOutputQueue(name=DISP_STREAM_NAME, 
                            maxSize=CAMERA_QUEUE_SIZE, 
                            blocking=QUEUE_BLOCKING)

    queueDepth = OAKD.getOutputQueue(
        name=DEPTH_STREAM_NAME,
        maxSize=CAMERA_QUEUE_SIZE,
        blocking=QUEUE_BLOCKING)

    while True:
        lastFrame = {}
        lastFrame["disp"] = queueStereo.tryGet()
        lastFrame["depth"] = queueDepth.tryGet()        

        if lastFrame["depth"] is not None:
            frameDepth = lastFrame["depth"].getFrame()
            depth_value_mm = frameDepth[mouseY, mouseX]

        if lastFrame["disp"] is not None:
            frameDisp = lastFrame["disp"].getFrame()
            maxDisparity = depth.initialConfig.getMaxDisparity()
            frameDisp = (frameDisp * 255. / maxDisparity).astype(numpy.uint8)
            frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)

            if len(frameDisp.shape) < 3:
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
            
            cv2.putText(frameDisp, f"Depth ({mouseX},{mouseY}): {depth_value_mm} mm", 
                            (0, 3 * TEXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            
            cv2.imshow(WINDOW_NAME, frameDisp)
            frameDisp = None
            frameDepth = None

        # Premere 'q' per uscire
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            exit()
            break