from pypylon import pylon 
import os, argparse, time
import logger
import utils
from cam_config import CamConfig
from sequencer import save_list, process_raw, add_arguments, parse_params

class AugoConfig(CamConfig):
    def __init__(self, save_root, cfgs={}):
        self.save_suffix = self.config_save_suffix(cfgs)
        self.save_dir = save_root + self.save_suffix

        self.log = logger.Logger(self.save_dir)
        self.cfgs = cfgs
        self.cam_cfg = {} # record camera configuration

        self.cam = self.get_device()
        self.config_camera(cfgs)

        self.disable_light_preset(cfgs, self.cam)
        self.config_auto_func(self.cam, cfgs)

    def config_save_suffix(self, cfgs):
        suffix = cfgs['suffix']
        return suffix

    def config_auto_func(self, cam, cfgs):
        print('Setting ExposureAuto')
        cam.ExposureAuto = "Continuous"

        print('Setting GainAuto')
        cam.GainAuto = "Continuous"

        print('Setting BalanceWhiteAuto')
        cam.BalanceWhiteAuto = "Continuous"
        cam.AutoTargetBrightness = cfgs.get('auto_target', 0.2)
        print('Setting AutoTargetBrightness: %.5f' % cam.AutoTargetBrightness.Value)
        cam.AutoFunctionProfile = 'MinimizeGain'

        cam.AutoGainUpperLimit = 12
        print('AutoGainUpperLimit: %.2f' % cam.AutoGainUpperLimit.Value)

        cam.AutoExposureTimeUpperLimit = int(1.0 / (2 * cfgs['fps']) * 1e6) # in us
        print('AutoExposureTimeUpperLimit: %.1f' % cam.AutoExposureTimeUpperLimit.Value)

        cam.ChunkModeActive = True
        for selector in ["ExposureTime", "Gain"]:
            print("Enable Chunkmode for %s" % (selector))
            cam.ChunkSelector = selector; 
            cam.ChunkEnable = True

def config_save_root(cfgs):
    cpath = os.path.splitext(__file__)[0] # script name
    save_root = os.path.join('results', '%s_%s' % (utils.get_datetime(), os.path.basename(cpath)))
    return save_root

def main(cfgs):
    save_root = config_save_root(cfgs)
    cam_agent = AugoConfig(save_root, cfgs)
    save_dir = cam_agent.get_save_dir()

    cam_agent.log.print_write('Start Grabbing, Sleep %f for buffering' % cfgs['sleep_t'])
    time.sleep(cfgs['sleep_t'])

    cam = cam_agent.cam
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
    cam_agent.log.reset_timer()

    img_list, expo_list = [], []

    count = 0 # number of captured images

    keep_capture = True
    while keep_capture:
        res = cam.RetrieveResult(cfgs['timeoutms'])
        if not res.GrabSucceeded():
            raise Exception("Grab Unsuccessful")
        
        exp_time = res.ChunkDataNodeMap.ChunkExposureTime.Value
        gain = res.ChunkDataNodeMap.ChunkGain.Value
        im = res.Array
        print('Exposures: %.2f, Gain: %.2f, WB: %s' % (exp_time, gain, cam_agent.get_wb_ratio(cam)))

        count += 1 

        if cam_agent.log.get_elapsed_seconds() >= cfgs['sec'] or\
            (cfgs['max_num'] > 0 and count >= cfgs['max_num']):
            keep_capture = False

            if cfgs['save_data'] :
                img_name = cam_agent.save_img(save_dir, im, count=count, pixelfmt=cfgs['pixelf'])
                img_list.append(img_name)
                expo_list.append(exp_time)

        res.Release()
    
    cam.StopGrabbing()
    cam_agent.record_metadata(cam) # record current camera meta data to cam_cfgs
    cam_agent.save_meta_data() # save cam_cfgs
    cam_agent.save_key_meta_data(cam) # save wb, exposure, and gain

    cam.ExposureAuto = "Off"
    cam.GainAuto = "Off"
    cam.Close()

    img_list_name = 'img_list.txt'
    img_list, expo_list = [img_list[-1]], [expo_list[-1]]
    save_list(save_dir, save_name=img_list_name, img_list=img_list, expo_list=expo_list)

    if cfgs['save_data'] and 'Bayer' in cfgs['pixelf'] and (cfgs['merge_hdr'] or cfgs['process_raw']):
        process_raw(cfgs, save_dir, img_list_name, save_suffix='')

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser = add_arguments(parser)
    parser.set_defaults(max_num=30, sec=5, do_wb=False, merge_hdr=False)
    args = parser.parse_args()

    cfgs = vars(args)
    parse_params(cfgs, cfgs['params']) # parse --params
    main(cfgs)

