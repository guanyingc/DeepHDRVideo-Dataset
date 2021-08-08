from pypylon import pylon
import numpy as np
import os, time
import cv2
import logger

# BalanceRatio, BalanceWhiteAuto, ColorTransformationValue, ColorTransformationSelector, DemosaicingMode
class CamConfig(object):
    def __init__(self, save_root, cfgs={}):

        self.save_suffix = self.config_save_suffix(cfgs)
        self.save_dir = save_root + self.save_suffix
        print(save_root, self.save_dir)

        self.log = logger.Logger(self.save_dir)
        self.log.print_params(cfgs)

        self.cfgs = cfgs
        self.cam_cfg = {} # record camera configuration

        self.cam = self.get_device()
        self.config_camera(cfgs)

        self.record_metadata(self.cam) # record current camera meta data to cam_cfgs

        if cfgs['load_wb']:
            self.load_wb(self.cam, cfgs['mdata_path']) # Only overwrite wb0 in cam_cfgs
            # Note that this white balance values are calibrated using auto_wb_gain.py and stored in camera/meta_data.npy


        self.disable_light_preset(cfgs, self.cam) # reset camera meta data

        self.config_sequencer(cfgs)

        self.save_meta_data('cam_data.npy') # save cam_cfgs

    def record_metadata(self, cam):
        ccm0 = self.get_color_correction_matrix(cam)
        wb0 = self.get_wb_ratio(cam)
        black_level = cam.BlackLevel.GetValue()
        gain = cam.Gain.GetValue()
        exposure = cam.ExposureTime.GetValue()
        self.cam_cfg.update({'ccm0': ccm0, 'wb0': wb0, 
            'gain': gain, 'blackl': black_level, 'exposure': exposure})

    def save_meta_data(self, save_name=''):
        if save_name == '':
            save_name = 'cam_data.npy'

        for key in self.cam_cfg:
            print('[Cam]: %s' % (key), self.cam_cfg[key])
        np.save(os.path.join(self.save_dir, save_name), self.cam_cfg)

    def save_key_meta_data(self, cam, save_name='meta_data.npy'):
        meta_data = {}
        for key in ['gain', 'wb0', 'exposure']:
            meta_data[key] = self.cam_cfg[key]
        np.save(save_name, meta_data)

    def load_wb(self, cam, meta_data_path):
        meta_data = np.load(meta_data_path, allow_pickle=True)[()]
        for key in ['wb0']:
            print("Overwriting %s: %s" % (key, meta_data[key]))
            self.cam_cfg[key] = meta_data[key]

        #for key, val in meta_data.items():
        #    print("Overwriting %s: %s" % (key, val))
        #    self.cam_cfg[key] = val

    def parse_params(self, cfgs, params):
        int_params = list(map(int, params.split(',')))
        print(int_params)
        cfgs['base_expo'] = int_params[0]
        cfgs['stop'] = int_params[1]
        cfgs['expo_n'] = int_params[2]
        if len(int_params) > 3:
            cfgs['max_num'] = int_params[3]
        if len(int_params) > 4:
            cfgs['sec'] = int_params[4]

    def config_save_suffix(self, cfgs):
        suffix = '_%dstop_%dexps_%ds' % (cfgs['stop'], cfgs['expo_n'], cfgs['sec'])
        if cfgs['max_num'] > 0:
            suffix += '_%dn' % cfgs['max_num']
        if cfgs['gain'] > 0:
            suffix += '_%.1fg' % cfgs['gain']
        if cfgs['suffix'] != '':
            suffix += '_%s' % cfgs['suffix']
        if cfgs['load_wb']:
            suffix += '_wb'
        return suffix

    def get_save_suffix(self):
        return self.save_suffix

    def get_save_dir(self):
        return self.save_dir

    def get_device(self):
        prompt = self.log.print_write
        info = pylon.DeviceInfo()
        info.SetDeviceClass("BaslerUsb")
        # open the first USB device
        cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))

        # this code only works for ace USB
        prompt("Found device %s " % cam.GetDeviceInfo().GetModelName())
        if not cam.GetDeviceInfo().GetModelName().startswith("acA"):
            prompt("_This_ sequencer configuration only works to basler ace USB")
        return cam

    def config_camera(self, cfgs={}):
        prompt = self.log.print_write
        cam = self.cam
        cam.Open() # open device
        cam.UserSetSelector = "Default"
        cam.UserSetLoad.Execute()

        # Set Height and width 
        width, height = cfgs.get('w', 4096), cfgs.get('h', 2168)
        cam.Width, cam.Height = width, height
        prompt('Setting image size to [%d, %d]' % (height, width))

        # Set fps
        fps = cfgs.get('fps', -1)
        if fps > 0:
            cam.AcquisitionFrameRateEnable = True
            cam.AcquisitionFrameRate = fps
            prompt('Setting FPS to %d' % fps)
        prompt('FPS: %f' % (cam.AcquisitionFrameRate.GetValue()))
        
        # Set pixel format
        pixel_format = cfgs.get('pixelf', 'BayerRG12')
        cam.PixelFormat = pixel_format
        prompt('Pixel format %s' % (pixel_format))
        self.cam_cfg.update({'w': cam.Width(), 'h': cam.Height(), 'fps': cam.AcquisitionFrameRate(),
                          'fmt': cam.PixelFormat()})
        
        # Set Gain
        gain = cfgs.get('gain', 0)
        cam.Gain = gain
        
        cam.MaxNumBuffer = 250

    def config_sequencer(self, cfgs={}):
        prompt = self.log.print_write
        cam = self.cam

        prompt('Setting chunk mode') # enable camera chunk mode
        cam.ChunkModeActive = True        # enable exposuretime chunk
        cam.ChunkSelector = "ExposureTime"
        cam.ChunkEnable = True

        prompt('Setting sequencer mode')
        cam.SequencerMode = "Off"
        cam.SequencerConfigurationMode = "On"

        base_expo = cfgs.get('base_expo', 2000) 
        stop = cfgs.get('stop', 3)
        expo_n = cfgs.get('expo_n', 2)
        expos = [base_expo] 
        self.cam_cfg.update({'expos': expos, 'base_expo': base_expo, 'stop': stop, 'expo_n': expo_n})

        for i in range(1, expo_n):
            expos += [expos[-1] * 2**stop]

        for i, expo in enumerate(expos):
            prompt('==> Expos %d: %f' % (i, expo))
            cam.SequencerSetSelector = i
            cam.ExposureTime = expo
            if i == len(expos) - 1: # The last set
                cam.SequencerSetNext = 0
            cam.SequencerSetSave.Execute()

        cam.SequencerConfigurationMode = "Off"
        cam.SequencerMode = "On"

        self.prev_expo = -1 # for check_exposure_order()

    def disable_light_preset(self, cfgs, cam):
        light_preset = cfgs.get('light_preset', False)

        if not light_preset:
            cam.LightSourcePreset = 'Off'
        ccm1 = self.get_color_correction_matrix(cam)
        wb1 = self.get_wb_ratio(cam)
        self.cam_cfg.update({'ccm1': ccm1, 'wb1': wb1})

    def get_color_correction_matrix(self, cam):
        # RGB
        ccm = []
        for key in ['Gain00', 'Gain01', 'Gain02',
                    'Gain10', 'Gain11', 'Gain12',
                    'Gain20', 'Gain21', 'Gain22']:
            cam.ColorTransformationValueSelector.SetValue(key)
            value = cam.ColorTransformationValue.GetValue()
            ccm.append(value)
        ccm = np.array(ccm).astype(np.float).reshape(3, 3)
        return ccm

    def get_wb_ratio(self, cam):
        # RGB gains
        balance_ratio = []
        #for i in ['Red', 'Green', 'Blue'] # BGR
        for channel in ['Red', 'Green', 'Blue']: # BGR
            cam.BalanceRatioSelector.SetValue(channel)
            balance_ratio.append(cam.BalanceRatio())
        balance_ratio = np.array(balance_ratio).astype(np.float)
        return balance_ratio
        
    def set_wb_ratio(self, cam, wb_rgb):
        # RGB gains
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            cam.BalanceRatioSelector.SetValue(channel)
            cam.BalanceRatio = wb_rgb[i]

    def check_exposure_order(self, res):
        exp_time = res.ChunkDataNodeMap.ChunkExposureTime.Value
        im = res.Array
        self.log.print_write("\t%d\t%6.0f: [%d, %d]" % \
                (res.BlockID, exp_time, im.shape[0], im.shape[1]))
        if exp_time == self.prev_expo:
            self.log.print_write('***** Invalid Time sequence ********')
        prev_expo = exp_time
        #print('prev_expo %f' % prev_expo)
        return exp_time

    def save_img(self, save_dir, im, count=0, pixelfmt='BayerRG12', ext='.jpg'):
        if pixelfmt in ['BGR8', 'RGB8']:
            img_name = '%s_%03d' % (pixelfmt, count) + '.jpg'
            cv2.imwrite(os.path.join(save_dir, img_name) , im)
        else:
            img_name = '%s_%03d' % (pixelfmt, count) + '.npy'
            np.save(os.path.join(save_dir, img_name) , im)
        return img_name

