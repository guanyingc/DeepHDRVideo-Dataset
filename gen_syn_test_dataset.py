import argparse
import numpy as np
import os
import glob
import utils
import logger
import image_utils as iutils
import cv2
np.random.seed(0)

class SynthVideoGenerator(object):
    def __init__(self, args):
        self.args = args
        args.suffix += '_%s' % utils.get_datetime(minutes=False)

        if args.expo_n == 2:
            args.stop_intv = 3
        elif args.expo_n == 3:
            args.stop_intv = 2

    def config_outdir(self, suffix=''):
        args = self.args
        suffix = self.config_suffix(suffix)
        dataset = os.path.basename(utils.remove_slash(args.in_dir) + '_%s' % suffix)
        args.out_dir = os.path.join(args.out_root, dataset)

        utils.make_file(args.out_dir)
        self.logger = logger.Logger(args.out_dir)
        self.logger.print_params(args)
        return args.out_dir

    def config_suffix(self, suffix):
        args = self.args
        if args.max_num > 0:
            suffix += '_%dn' % args.max_num
        if args.sample_step > 0:
            suffix += '_%dsample' % args.sample_step
        if args.resize:
            suffix += '_%d_%d' % (args.h, args.w)
        return suffix 

    def get_exposures(self, nframes, nexps, ref_hdr):
        if nexps == 2:
            base_expos = [1, 8]
        elif nexps == 3:
            base_expos = [1, 4, 16]
        expos = np.tile(base_expos, nframes//nexps+1)[:nframes]
        new_max_val = np.percentile(ref_hdr, 99.5) # This value will be used to normalize the HDR image
        return expos, new_max_val
    
    def save_base_expos(self, out_scene_dir, expos, nexps):
        base_expos = expos[:args.expo_n]
        if nexps == 2:
            base_expos = base_expos / 8.0
        elif nexps == 3:
            base_expos = base_expos / 4.0
        stops = np.log2(base_expos)
        utils.save_list(os.path.join(out_scene_dir, 'Exposures.txt'), ['%.2f' % stop for stop in stops])

    def check_scene_dir(self, scene, out_scene_dir, print_prefix):
        args = self.args

        # Get hdr image list, config out dir
        hdrs = self.get_hdr_list(os.path.join(args.in_dir, scene))

        if args.sample_step > 0:
            hdrs = [hdrs[i] for i in range(0, len(hdrs), args.sample_step)]

        img_list, hdr_list = [], []
        
        ref_hdr = iutils.read_hdr(hdrs[0])
        expos, new_max_val = self.get_exposures(len(hdrs), args.expo_n, ref_hdr)

        print(expos, new_max_val)
        self.save_base_expos(out_scene_dir, expos, args.expo_n)


        nframes = len(hdrs)
        if args.max_num > 0 and args.max_num < nframes:
            nframes = args.max_num

        for i_h in range(args.start_frame, nframes):
            
            if i_h % args.print_intv == 0:
                self.logger.print_write('\t %s [%d/%d] %s: ' % (print_prefix, i_h, nframes, hdrs[i_h]))

            hdr_name = os.path.splitext(os.path.basename(hdrs[i_h]))[0]
            save_name = os.path.join(out_scene_dir, hdr_name)

            hdr = iutils.read_hdr(hdrs[i_h])
            hdr = (hdr / new_max_val).clip(0, 1)

            if args.border > 0:
                hdr = iutils.crop_img_border(hdr, args.border)

            if args.resize:
                self.logger.print_write('Resizing')
                hdr = cv2.resize(hdr, (args.w, args.h))
            
            # For 2-exposure scene, the first and the last frames do not have the GT HDR
            # For 3-exposure scene, the first two and the last two frames do not have the GT HDR
            if (args.expo_n == 2 and 1 < i_h < nframes-2) or (args.expo_n == 3 and 2 < i_h < nframes-3):
                hdr_list.append('%s.hdr %f' % (os.path.basename(save_name), expos[i_h]))
                iutils.save_hdr(save_name + '.hdr', hdr.astype(np.float16))
            
            if expos[i_h] == 1.0 or expos[i_h] == 4.0:
                print('Adding noise for the low exposure or middle exposure images')
                hdr = self.add_noise(hdr, scale=expos[i_h])

            iutils.save_uint16(save_name + '.tif', iutils.hdr_to_ldr(hdr, expo=expos[i_h]))
            iutils.save_uint8(save_name + '.jpg', iutils.hdr_to_ldr(hdr, expo=expos[i_h]))  # for visualization
            iutils.save_uint8(save_name + 'adj.jpg', iutils.hdr_to_ldr(hdr, expo=max(expos))) # for visualization
            img_list.append(os.path.basename(save_name))

        utils.save_list(os.path.join(out_scene_dir, 'hdr_expos.txt'), hdr_list)
        utils.save_list(os.path.join(out_scene_dir, 'img_list.txt'), img_list)
        return img_list

    def add_noise(self, img, mean=0, stdv1=1e-3, stdv2=3e-3, scale=1):
        stdv = np.random.uniform(stdv1, stdv2)
        noise = np.random.normal(loc=mean, scale=stdv, size=img.shape) / scale
        img += noise
        return img.clip(0, 1)

    def get_hdr_list(self, in_dir, max_num=400):
        hdrs = sorted(glob.glob(os.path.join(in_dir, '*.exr')))
        if len(hdrs) >= max_num:
            idx = np.linspace(0, len(hdrs) - 1, max_num).astype(int)
            hdrs = [hdrs[i] for i in idx]
        return hdrs

    def print_write(self, strings):
        self.logger.print_write(strings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='.')
    parser.add_argument('--scene_list', default='scenes_CVPR2021.txt')
    parser.add_argument('--out_root', default='./data/')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--h', default=720, type=int)
    parser.add_argument('--w', default=1280, type=int)
    parser.add_argument('--expo_n', default=2, type=int) # number of exposures
    parser.add_argument('--max_num', default=60, type=int) # maximum number of frames
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--sample_step', default=-1, type=int) # interval between the sampled frames
    parser.add_argument('--print_intv', default=1, type=int)
    parser.add_argument('--border', default=15, type=int) # crop the dark border regions, 15 for HDM dataset
    parser.add_argument('--resize', default=False, action='store_true')
    parser.add_argument('--add_noise', default=True, action='store_false')
    args = parser.parse_args()

    generator = SynthVideoGenerator(args)
    generator.config_outdir(args.suffix) 

    scenes = np.genfromtxt(os.path.join(args.in_dir, args.scene_list), 'str')
    if scenes.ndim == 0: # only have one secene
        scenes = scenes.reshape(1)
    print(scenes)

    scene_list = []
    for i_s, scene in enumerate(scenes):
        print_prefix = '[%d/%d]' % (i_s + 1, len(scenes))
        generator.print_write('%s: %s/%s:' % (print_prefix, args.in_dir, scene))
        
        out_scene_name = '%s_%dexpo' % (scene, args.expo_n)
        out_scene_dir = os.path.join(args.out_dir, 'Images', out_scene_name)
        utils.make_file(out_scene_dir)
        utils.save_list(os.path.join(args.out_dir, '%s.txt' % out_scene_name), [out_scene_name])

        img_list = generator.check_scene_dir(scene, out_scene_dir, print_prefix)

        scene_list.append('%s' % (out_scene_name))
    utils.save_list(os.path.join(args.out_dir, 'scenes_%dexpo.txt' % args.expo_n), scene_list)
