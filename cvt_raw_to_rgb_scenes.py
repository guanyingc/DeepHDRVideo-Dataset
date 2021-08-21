"""
convert raw data to rgb image
"""
import argparse
import os
import glob
import utils
import cv2
import numpy as np
import image_utils as iutils
from digital_ISP import DigitalISP, HDRMerger


class RawDataProcessor(object):
    def __init__(self, args):
        self.args = args
        save_suffix = self.config_save_suffix(args)
        args.out_dir = self.config_outdir(args.out_root, args.in_dir, save_suffix)
    
    def config_save_suffix(self, args):
        suffix = '_%.3fr' % args.ratio
        return suffix

    def config_outdir(self, out_root, in_dir, save_suffix=''):
        out_dir = os.path.basename(utils.remove_slash(in_dir))
        out_dir += '_parsed_%s' % utils.get_datetime(minutes=False)
        if save_suffix != '':
            out_dir += save_suffix
        out_dir = os.path.join(out_root, out_dir) 
        utils.make_file(out_dir)
        return out_dir

    def process_scenes(self):
        args = self.args
        scenes = self.prepare_scene_list(args.in_dir, args.scene_list)
        if len(scenes) == 0:
            raise Exception('Scene number is 0!')

        utils.save_list(os.path.join(args.out_dir, 'scene_all.txt'), scenes)

        for i_s in range(args.start_frame, len(scenes)):
            scene = scenes[i_s]
            print('[%d/%d] Processing Scene: %s' % (i_s+1, len(scenes), scene))
            self.process_scene(args, scenes, i_s)

    def prepare_scene_list(self, in_dir, scene_list):
        scene_list = os.path.join(in_dir, scene_list)
        if os.path.exists(scene_list):
            print('Loading Scene list: %s' % scene_list)
            scenes = utils.read_list(scene_list)
        else:
            print('Glob scene list: %s' % scene_list)
            files = sorted(glob.glob(os.path.join(in_dir, '*')))
            scenes = [os.path.basename(file_name) for file_name in files if os.path.isdir(file_name)] # filter no directory file
        return scenes

    def process_scene(self, args, scenes, scene_idx):
        scene = scenes[scene_idx]
        scene_dir, img_paths, expos, cam_data, isp_cfgs = self.load_scene_data(args, scene)
        digital_isp = DigitalISP(isp_cfgs)
        digital_isp.set_meta_data(cam_data)

        save_dir = os.path.join(args.out_dir, scene)
        utils.make_file(save_dir)
        save_img_list = []

        if args.merge_hdr:
            imgs = []
        for i, img_path in enumerate(img_paths):

            if args.max_num > 0 and i > args.max_num: # for debuging
                break

            if i % 5 == 0:
                print('\t %d/%d scene: %d/%d Processing %s' % (
                    scene_idx+1, len(scenes), i, len(img_paths), img_path))

            img = digital_isp.camera_isp(np.load(img_path))

            if args.merge_hdr: # cache for hdr reconstruction
                imgs.append(img)

            if args.max_frame_num > 0 and i >= args.max_frame_num:
                print('[scene %d] Max frame number: %d, skip %d image' % (scene_idx, args.max_frame_num, i+1))
            else:
                img_name = 'img_%03d' % i
                save_name = '%s%s' % (img_name, args.ext)

                save_img_list.append(save_name)
                if args.ext == '.tif':
                    iutils.save_uint16(os.path.join(save_dir, save_name), img)
                else:
                    iutils.save_uint8(os.path.join(save_dir, save_name), img)

        exposures = np.log2(expos[:len(np.unique(expos))])

        utils.save_list(os.path.join(save_dir, 'Exposures.txt'), exposures) # save exposure
        utils.save_list(os.path.join(save_dir, 'img_list.txt'), save_img_list)

        if args.merge_hdr:
            hdr_merger = HDRMerger()
            hdr = hdr_merger.merge_hdrs(imgs, expos)
            print('HDR size: %s' % str(hdr.shape))

            hdr_name = '%s.hdr' % scene
            iutils.save_hdr(os.path.join(save_dir, hdr_name), hdr)
            iutils.save_uint8(os.path.join(save_dir, '%s_loghdr.jpg' % scene), iutils.mulog_transform(hdr))
            iutils.save_uint8(os.path.join(args.out_dir, '%s_loghdr.jpg' % scene), iutils.mulog_transform(hdr))

            save_img_hdr_list = ['%s %s' % (img_name, hdr_name) for img_name in save_img_list]
            utils.save_list(os.path.join(save_dir, 'img_hdr_list.txt'), save_img_hdr_list)

    def load_scene_data(self, args, scene):
        scene_dir = os.path.join(args.in_dir, scene)
        img_name_expos = np.genfromtxt(os.path.join(scene_dir, args.img_list), dtype='str')
        img_paths = [os.path.join(scene_dir, img_name) for img_name in img_name_expos[:, 0]]
        expos = img_name_expos[:, 1].astype(float)
        expos = expos / expos.min()

        cam_data = np.load(os.path.join(scene_dir, 'cam_data.npy'), allow_pickle=True)[()]

        cvtfmt = cv2.COLOR_BAYER_RG2RGB #cvtfmt = cv2.COLOR_BAYER_BG2BGR_EA
        isp_cfgs = {'ratio': args.ratio, 'bit': 12, 'wb': cam_data['wb0'], 
                'gamma': True, 'cvtfmt': cvtfmt, 'anti_alias': True, 'do_ccm': True}
        return scene_dir, img_paths, expos, cam_data, isp_cfgs


def add_arguments(parser):
    parser.add_argument('--in_dir', default='')
    parser.add_argument('--out_root', default='./real_data/')
    parser.add_argument('--scene_list', default='scenes.txt')
    parser.add_argument('--img_list', default='img_list.txt')
    parser.add_argument('--ratio',default=0.375, type=float)
    parser.add_argument('--max_num', default=-1, type=int)
    parser.add_argument('--max_frame_num', default=-1, type=int)
    parser.add_argument('--ext', default='.tif')
    parser.add_argument('--start_frame',default=0,type=int)
    parser.add_argument('--merge_hdr', default=True, action='store_false')
    return parser


def main(args):
    processer = RawDataProcessor(args)
    processer.process_scenes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)
