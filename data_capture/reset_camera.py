from pypylon import pylon 

def get_device():
    info = pylon.DeviceInfo()
    info.SetDeviceClass("BaslerUsb")

    # open the first USB device
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))

    # this code only works for ace USB
    print("Found device %s " % cam.GetDeviceInfo().GetModelName())
    if not cam.GetDeviceInfo().GetModelName().startswith("acA"):
        print("_This_ sequencer configuration only works to basler ace USB")
    return cam

def main():
    cam = get_device()
    cam.Open() # open device
    cam.UserSetSelector = "Default"
    cam.UserSetLoad.Execute()
    cam.Close()

if __name__ == "__main__":
    main()
    
