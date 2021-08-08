from pypylon import pylon 
import numpy as np
import os, argparse, time
import cv2
import sys
sys.path.insert(0, ".")
import utils
from cam_config import CamConfig
from digital_ISP import DigitalISP

# global variables that can be accessed by SampleImageEventHandler
img_list, expo_list = [], []
count = 0
save_dir = ''
pixelfmt = ''
save_data = True

class SampleImageEventHandler(pylon.ImageEventHandler):

    def OnImageGrabbed(self, camera, res):
        pixelf = 'BayerRG12'

        exp_time = res.ChunkDataNodeMap.ChunkExposureTime.Value
        
        # For debug purporse, print if current exposure == previous exposure
        if not hasattr(self, 'prev_expo'):
            self.prev_expo = -1

        if exp_time == self.prev_expo:
            print('***** Invalid Time sequence ********')
        self.prev_expo = exp_time

        im = res.Array
        print("\t%d\t%6.0f: [%d, %d]" % (res.BlockID, exp_time, im.shape[0], im.shape[1]))
        
        if save_data:
            img_name = '%s_%03d' % (args.pixelf, res.BlockID) + '.npy'
            np.save(os.path.join(save_dir, img_name) , im)
            img_list.append(img_name)
            expo_list.append(exp_time)

    def OnImagesSkipped(self, camera, skip_num):
        print('OnImagesSkipped: skip %d images' % (skip_num))

def set_img_event_handler(cam):
    cam.Close()
    cam.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), 
            pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)
    cam.RegisterImageEventHandler(SampleImageEventHandler(), 
            pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
    cam.MaxNumBuffer = 250
    cam.Open()

def config_save_root(cfgs, cpath='.'):
    if cfgs['max_num'] < 0 or cfgs['max_num'] > 5:
        datetime = utils.get_datetime(minutes=True, seconds=True)
    else:
        datetime = utils.get_datetime()
    save_root = os.path.join('results', '%s_%s' % (datetime, os.path.basename(cpath))) 
    return save_root

def countdown(t, step=1, msg='sleeping'):  # in seconds
    for remaining in range(t, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining.".format(remaining))
        sys.stdout.flush()
        time.sleep(1)
    print('')
    print('\n****Sleep done for %d seconds!****' % (t))

def save_list(save_dir, save_name, img_list, expo_list):
    img_expo_list = ['%s %f' % (img, expo) for img, expo in zip(img_list, expo_list)]
    utils.save_list(os.path.join(save_dir, save_name), img_expo_list)

def process_raw(cfgs, save_dir, img_list_name, save_suffix=''):
    isp_cfgs = { 'bit': 12, 'ratio': cfgs['ratio'], 'ext': cfgs['img_ext'], 
            'do_wb': cfgs['do_wb'], 'anti_alias': False}
    digital_isp = DigitalISP(isp_cfgs, mdata_path=os.path.join(save_dir, 'cam_data.npy'),
            img_dir=save_dir, img_list=img_list_name)

    if cfgs['process_raw']:
        rgb_save_dir = save_dir
        if save_suffix != '': 
            sub_dir = 'RGB_%.2fr%s' % (cfgs['ratio'], save_suffix)
            rgb_save_dir = os.path.join(rgb_save_dir, sub_dir)
        utils.make_file(rgb_save_dir)
        print('RGB Save dir: %s' % rgb_save_dir)
        digital_isp.cvt_raw_to_rgb(rgb_save_dir, merge_hdr=cfgs['merge_hdr'])

    if not cfgs['process_raw'] and cfgs['merge_hdr']: # only do merge HDR
        print('Merging HDR')
        hdr_save_dir = os.path.join(save_dir)
        digital_isp.merge_hdrs(hdr_save_dir)

def add_arguments(parser):
    # Capture options
    parser.add_argument('--h', default=2168, type=int) # 720
    parser.add_argument('--w', default=4096, type=int) # 1280
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('-b', '--base_expo', default=1000, type=int) # base exposures
    parser.add_argument('-e', '--expo_n', default=2, type=int) # number of exposures
    parser.add_argument('-s', '--stop', default=3, type=float) # stop
    parser.add_argument('-m', '--max_num', default=-1, type=int) # maximum number of frames
    parser.add_argument('--sec', default=1, type=float) # capture videos with n seconds
    parser.add_argument('-p', '--params', default='') # all params, 'base_exp,stop,exp_n,max_num,sec'
    parser.add_argument('-g', '--gain', default=0, type=float) # gain

    parser.add_argument('--timeoutms', default=1000, type=int) # time out in ms for waiting
    parser.add_argument('-t', '--sleep_t', default=0.2, type=float) # sleep time before retrive images
    parser.add_argument('--pixelf', default='BayerRG12p', help='BayerRG12|BayerRG12p|BayerRG8|BGR8|Mono8|YCbCr422_8')
    parser.add_argument('-a', '--auto_target', default=0.2, type=float) # base exposures
    
    parser.add_argument('--load_wb', default=True, action='store_false')
    parser.add_argument('--mdata_path', default='meta_data.npy', help='path to gain') # metadata path
    
    # Process image options
    parser.add_argument('--save_data', default=True, action='store_false')
    parser.add_argument('--ratio', default=0.375, type=float) # rescale ratio
    parser.add_argument('--process_raw',default=True, action='store_false')
    parser.add_argument('--img_ext', default='jpg', help='jpg|png|tif')
    parser.add_argument('--merge_hdr', default=True, action='store_false')
    parser.add_argument('--mute', default=True, action='store_false')
    parser.add_argument('--do_wb', default=True, action='store_false')
    parser.add_argument('--suffix', default='') # sleep time before retrive images
    return parser

def main(cfgs):
    cpath = os.path.splitext(__file__)[0] # script name
    save_root = config_save_root(cfgs, cpath)
    cam_agent = CamConfig(save_root, cfgs)
    global save_dir
    save_dir = cam_agent.get_save_dir()

    cam = cam_agent.cam
    set_img_event_handler(cam)
    global pixelfmt, save_data
    pixelfmt = cfgs['pixelf']
    save_data = cfgs['save_data']

    if cfgs['sleep_t'] > 0:
        countdown(int(cfgs['sleep_t']))
    cam_agent.log.print_write('Start Grabbing, Sleep %d for buffering' % cfgs['sleep_t'])
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
    cam_agent.log.reset_timer()

    global count
    while True:
        if cam.WaitForFrameTriggerReady(cfgs['timeoutms'], pylon.TimeoutHandling_ThrowException):
            cam.ExecuteSoftwareTrigger();

        count = count + 1 

        if cam_agent.log.get_elapsed_seconds() >= cfgs['sec'] or \
                (cfgs['max_num'] > 0 and count >= cfgs['max_num']):
            break

    print('Trigger finished: Queued: %d, Ready: %d, Output: %d' % \
            (cam.NumQueuedBuffers.Value, cam.NumReadyBuffers.Value, cam.OutputQueueSize.Value))
    time.sleep(0.4)
    while (cam.NumReadyBuffers.Value > 0):
        time.sleep(0.1)

    cam.StopGrabbing()
    cam.SequencerMode = "Off"
    cam.Close()

    img_list_name = 'img_list.txt'
    save_list(save_dir, save_name=img_list_name, img_list=img_list, expo_list=expo_list)

    if cfgs['save_data'] and 'Bayer' in cfgs['pixelf'] and (cfgs['merge_hdr'] or cfgs['process_raw']):
        process_raw(cfgs, save_dir, img_list_name, save_suffix='')

def parse_params(cfgs, params):
    if params == '':
        return
    int_params = list(map(int, params.split(',')))
    print(int_params)

    cfgs['base_expo'] = int_params[0]
    cfgs['stop'] = int_params[1]
    cfgs['expo_n'] = int_params[2]

    if len(int_params) > 3:
        cfgs['max_num'] = int_params[3]
    if len(int_params) > 4:
        cfgs['sec'] = int_params[4]

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser = add_arguments(parser)
    args = parser.parse_args()

    cfgs = vars(args)
    parse_params(cfgs, cfgs['params']) # parse --params

    # Limit
    # 2exposure: max 1500, 3exposure max 1000

    if cfgs['expo_n'] == 2:
        assert(cfgs['base_expo'] <= 1500) 

    if cfgs['expo_n'] == 3:
        assert(cfgs['base_expo'] <= 1000) 
    
    # if the duration of the captured video is larger than 2 seconds, do not merge hdr
    #if cfgs['sec'] > 2:
    #    cfgs['merge_hdr'] = False

    main(cfgs)
