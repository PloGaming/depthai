import depthai as dai

with dai.Device() as device:
    for cam in device.getConnectedCameraFeatures():
        print(cam)
        #continue  # uncomment for less verbosity
        for cfg in cam.configs:
            print("   ", cfg)