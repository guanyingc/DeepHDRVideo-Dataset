import argparse
import os
import glob
import cv2
import utils
import numpy as np


class RealPatchProcessor(object):
    def __init__(self, args):
        self.args = args
        save_suffix = self.config_save_suffix(args)
        args.out_dir = self.config_outdir(args.out_root, args.in_dir, save_suffix)
    
    def config_save_suffix(self, args):
        suffix = ''
        return suffix

    def config_outdir(self, out_root, in_dir, save_suffix=''):
        out_dir = os.path.basename(utils.remove_slash(in_dir))
        out_dir += '_rand_motion_%s' % utils.get_datetime(minutes=False)
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
        
        for i_s in range(args.start_scene, len(scenes)):
            if args.max_scene_num > 0 and i_s > args.max_scene_num:
                break
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
        scene_dir = os.path.join(args.in_dir, scene)
        save_dir = os.path.join(args.out_dir, scene)
        utils.make_file(save_dir)

        img_names, imgs, expos, hdr = self.load_scene_data(scene_dir)

        rand_offsets = self.prepare_random_motion_offset(len(imgs), args.max_disp)

        new_imgs, new_hdrs = self.translate_imgs(imgs, hdr, rand_offsets, args.max_disp)
        
        img_list, img_hdr_list = [], []
        for i, img_name in enumerate(img_names):
            hdr_name = img_name[:-4] + '.hdr'
            img_list.append(img_name)
            img_hdr_list.append('%s %s' % (img_name, hdr_name))

            cv2.imwrite(os.path.join(save_dir, img_name), new_imgs[i])
            cv2.imwrite(os.path.join(save_dir, img_name[:-4]+'.jpg'), new_imgs[i]/65535.0*255)

            cv2.imwrite(os.path.join(save_dir, hdr_name), new_hdrs[i])


        utils.save_list(os.path.join(save_dir, 'img_list.txt'), img_list)
        utils.save_list(os.path.join(save_dir, 'img_hdr_list.txt'), img_hdr_list)

        center_expo = expos[len(expos)//2]
        exposures = ['%.2f' % (expo - center_expo) for expo in expos]
        utils.save_list(os.path.join(save_dir, 'Exposures.txt'), exposures)

    def translate_imgs(self, imgs, hdr, offsets, max_disp): 
        new_imgs = []
        new_hdrs = []
        for i, img in enumerate(imgs):
            new_img = self.translate(img, offsets[i])
            new_img = self.crop_border(new_img, border=max_disp)
            new_imgs.append(new_img)

            new_hdr = self.translate(hdr, offsets[i])
            new_hdr = self.crop_border(new_hdr, border=max_disp)
            new_hdrs.append(new_hdr)
        return new_imgs, new_hdrs
            
    def translate(self, img, offsets):
        disp_w, disp_h = offsets
        T = np.float32([[1, 0, disp_w], [0, 1, disp_h]]) 
        h, w, _ = img.shape
        img_translated = cv2.warpAffine(img, T, (w, h)) 
        return img_translated

    def load_scene_data(self, scene_dir):
        img_hdr_list = np.genfromtxt(os.path.join(scene_dir, args.img_hdr_list), 'str')
        img_names = img_hdr_list[:, 0]
        hdr_name = img_hdr_list[0, 1]
        
        imgs = []
        for img_name in img_names:
            img = cv2.imread(os.path.join(scene_dir, img_name), -1)
            imgs.append(img)
        hdr = cv2.imread(os.path.join(scene_dir, hdr_name), -1)
        expos = np.genfromtxt(os.path.join(scene_dir, 'Exposures.txt'))
        return img_names, imgs, expos, hdr

    def prepare_random_motion_offset(self, img_num, max_disp=20):
        start_x, start_y = 0, 0
        end_x, end_y = 0, 0
        while (np.abs(end_x) + np.abs(end_y)) < 2*max_disp*0.9: # threshold
            end_x = np.random.randint(-max_disp, max_disp)
            end_y = np.random.randint(-max_disp, max_disp)
        print('Start and end:', start_x, start_y, end_x, end_y)
        x_offsets = np.linspace(start_x, end_x, img_num).astype(int)
        y_offsets = np.linspace(start_y, end_y, img_num).astype(int)
        print('x-offsets', x_offsets)
        print('y-offsets', y_offsets)
        offsets = np.stack([x_offsets, y_offsets], 1)
        #print(offsets)
        return offsets

    def crop_border(self, img, border=20):
        img = img[border: -border, border: -border]
        return img


def add_arguments(parser):
    parser.add_argument('--in_dir', default='')
    parser.add_argument('--out_root', default='real_data')
    parser.add_argument('--scene_list', default='scene_all.txt')
    parser.add_argument('--img_hdr_list', default='img_hdr_list.txt')
    parser.add_argument('--start_scene', default=0, type=int)

    parser.add_argument('--max_scene_num', default=-1, type=int)
    parser.add_argument('--max_disp', default=30, type=int)
    return parser


def main(args):
    processer = RealPatchProcessor(args)
    processer.process_scenes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)

