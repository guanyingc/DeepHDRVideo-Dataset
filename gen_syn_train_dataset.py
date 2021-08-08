"""
Scripts for prepare synthetic training data
"""
import argparse
import os
import glob
import utils
import logger
import cv2
import numpy as np
import image_utils as iutils
np.random.seed(0)


class SynthGenerator(object):
    def __init__(self, args):
        self.args = args
        args.suffix += '_%s' % utils.get_datetime(minutes=False)
    
    def config_outdir(self, suffix=''):
        args = self.args
        save_dir = os.path.basename(utils.remove_slash(args.in_dir) + args.suffix + args.ext)

        args.out_dir = os.path.join(args.out_root, save_dir)
        utils.make_file(args.out_dir)

        self.logger = logger.Logger(args.out_dir)
        self.logger.print_params(args)
        return args.out_dir

    def get_scene_expos(self, scene_list):
        scene_expos = np.genfromtxt(scene_list, dtype='str')
        if scene_expos.ndim == 0:
            scene_expos = scene_expos.reshape(1, 1)
        elif scene_expos.ndim == 1: # if only have one scene
            scene_expos = scene_expos.reshape(1, -1)
        scenes = scene_expos[:, 0]
        """
        Important: min_percents indicate the minimum percent of well-exposed pixels
        in an image when synthesize the sequences with alternating exposures
        However, it is not used and can be safely ignored.
        """ 
        if scene_expos.shape[1] == 1:  
            # If do not sepecify the minimum exposure, set 99.99% as default
            min_percents = np.ones(len(scene_expos)) * 99.99
        else:
            min_percents = scene_expos[:,1].astype(np.float)
        base_expos = np.ones(min_percents.shape)
        return scenes, min_percents, base_expos

    def process_scene(self, scene, min_percent, base_expo, print_prefix):
        args = self.args

        # Step 1: Get hdr image list, config out dir
        hdrs = self.get_hdr_list(os.path.join(args.in_dir, scene))

        if scene == 'water': # This scene has repetitive frames, skip every 2 frames
            hdrs = hdrs[::2]

        out_scene_dir = os.path.join(args.out_dir, 'Images', scene)
        utils.make_file(out_scene_dir)

        max_boxes = args.max_boxes if args.max_boxes > 0 else int(np.ceil(1000. / len(hdrs)))

        # Step 2: Load triples and process
        patch_lists = []
        for i_h in range(args.start_frame, len(hdrs) - 3):
            verbose = False

            if args.max_num > 0 and i_h - args.start_frame > args.max_num: 
                break

            if i_h % args.print_intv == 0:
                self.logger.print_write('\t %s [%d/%d] %s: ' % (print_prefix, i_h, len(hdrs), hdrs[i_h]))
                verbose = True

            hdr_imgs = []
            if i_h == args.start_frame:
                for hdr_path in hdrs[i_h-3:i_h+4]:
                    hdr_imgs.append(self.read_process_hdr(hdr_path))
            else:
                for hdr in prev_hdr_imgs[1:]:
                    hdr_imgs.append(hdr.copy())
                hdr_imgs.append(self.read_process_hdr(hdrs[i_h+3]))

            hdr_name = os.path.splitext(os.path.basename(hdrs[i_h]))[0]
            patch_list, boxes = self.process_nframes(hdr_imgs,
                min_percent, base_expo, hdr_name, max_boxes, out_scene_dir, verbose=verbose)

            patch_lists += patch_list
            prev_hdr_imgs = hdr_imgs # cached data

        utils.save_list(os.path.join(out_scene_dir, scene + '_list.txt'), patch_lists)
        return patch_lists

    def get_hdr_list(self, in_dir, max_num=400):
        hdrs = sorted(glob.glob(os.path.join(in_dir, '*.exr')))
        if len(hdrs) >= max_num:
            idx = np.linspace(0, len(hdrs) - 1, max_num).astype(int)
            hdrs = [hdrs[i] for i in idx]
        return hdrs

    def read_process_hdr(self, hdr_path):
        hdr = iutils.read_hdr(hdr_path)

        if self.args.border > 0:
            hdr = iutils.crop_img_border(hdr, self.args.border)

        if self.args.resize:
            hdr = cv2.resize(hdr, (args.w, args.h))
        return hdr

    def process_nframes(self, hdr_imgs, min_percent, base_expo, hdr_name, max_boxes, out_scene_dir, fixed_p=-1, boxes=None, verbose=False):
        args = self.args

        mid = len(hdr_imgs) // 2
        cur_max = hdr_imgs[mid].max()

        n_hdr_imgs = []
        for hdr in hdr_imgs:
            n_hdr_imgs += [(hdr / cur_max).clip(0, 1)]

        h, w, c = n_hdr_imgs[mid].shape
        l_expo, h_expo = base_expo, base_expo * np.power(2.0, args.stop_intv)

        # Generate boxes for cropping patches
        if boxes is None:
            high_expo_prev = iutils.hdr_to_ldr(n_hdr_imgs[mid-1], expo=h_expo)
            high_expo_cur = iutils.hdr_to_ldr(n_hdr_imgs[mid], expo=h_expo)
            boxes = self.prepare_boxes(high_expo_prev, high_expo_cur, max_boxes)
        else:
            self.logger.print_write('** Use provided boxes  **')

        # Sample regions and save LDRs
        patch_list = []
        for i_b, box in enumerate(boxes):
            save_dir = os.path.join(out_scene_dir, '%s_%d' % (hdr_name, i_b))
            utils.make_file(save_dir)

            t, b, l, r = box
            hdr_list = []
            for suffix, hdr in zip(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'], n_hdr_imgs):

                # Save HDR
                hdr_crop = hdr[t:b, l:r]
                save_name = os.path.join(save_dir, os.path.basename(save_dir) + '_' + suffix)
                np.save(save_name, hdr_crop.astype(np.float16))
                hdr_list.append('%s %.6f' % (os.path.basename(save_name), min_percent))

                # Save LDR for visualization purpose
                iutils.save_uint8(save_name + '_' + suffix + '.jpg', iutils.hdr_to_ldr(hdr_crop, expo=h_expo))
            
            utils.save_list(os.path.join(save_dir, os.path.basename(save_dir) + '.txt'), hdr_list)
            scene = os.path.basename(out_scene_dir)
            patch_list.append(os.path.join(scene, os.path.basename(save_dir)))
        return patch_list, boxes

    def prepare_boxes(self, img1, img2, max_boxes):
        boxes = self.sort_motion_regions(img1, img2)
        string = 'Find %d boxes' % (len(boxes))
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
            string += ', Sample %d boxes' % (len(boxes))
        self.logger.print_write(string)
        return boxes

    def sort_motion_regions(self, img1, img2, bsize=352, stride=176):
        height, width, c = img1.shape
        h_idx = range(0, height - bsize, stride)
        w_idx = range(0, width - bsize, stride)
        box_scores = []
        for h in h_idx:
            for w in w_idx:
                t, b, l, r = h, h + bsize, w, w + bsize
                abs_diff = abs(img1[t: b, l: r] - img2[t: b, l: r]).mean()
                box_scores.append([abs_diff, [t, b, l, r]])
        box_scores.sort(reverse=True)
        boxes = [box_score[1] for box_score in box_scores]
        return boxes

    def print_write(self, strings):
        self.logger.print_write(strings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',     default='.')
    parser.add_argument('--scene_list', default='scenes.txt')
    parser.add_argument('--out_root',   default='./data/')
    parser.add_argument('--suffix',     default='_7f_HDR')
    parser.add_argument('--ext',        default='npy')
    parser.add_argument('--max_num',    default=-1, type=int) # for testing puporse
    parser.add_argument('--start_frame',default=3,  type=int)
    parser.add_argument('--stop_intv',  default=3,  type=int)
    parser.add_argument('--print_intv', default=1,  type=int)
    parser.add_argument('--frame_step', default=5,  type=int) # interval between the sampled frames
    parser.add_argument('--max_boxes',  default=-1, type=int) # max number of sampled patches in each image
    parser.add_argument('--border',     default=0, type=int)
    parser.add_argument('--resize',     default=False, action='store_true')
    parser.add_argument('--h',          default=720, type=int)
    parser.add_argument('--w',          default=1280, type=int)
    args = parser.parse_args()

    """Crop border for HdM-HDR-2014_Original-HDR-Camera-Footage dataset"""
    if "HdM-HDR-2014_Original-HDR-Camera-Footage" in args.in_dir:
        args.border = 10
        args.resize = True
        args.suffix += '_%d_%d' % (args.h, args.w)

    generator = SynthGenerator(args)
    generator.config_outdir(args.suffix) 

    scenes, min_percents, base_expos = generator.get_scene_expos(os.path.join(args.in_dir, args.scene_list))
    patch_list = []
    for i_s, scene in enumerate(scenes):
        print_prefix = '[%d/%d]' % (i_s + 1, len(scenes))
        generator.print_write('%s: %s/%s: base expo %.6f' % (
            print_prefix, args.in_dir, scene, base_expos[i_s]))
        scene_list = generator.process_scene(scene, min_percents[i_s], base_expos[i_s], print_prefix)
        patch_list += scene_list
    generator.prepare_list(patch_list)
