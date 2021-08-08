"""
helper functions for processing raw data and generate HDR from images
"""
import argparse
import os
import cv2
import utils
from imageio import imread, imsave
import numpy as np


class DigitalISP(object):
    def __init__(self, cfgs, mdata_path='', img_dir='', img_list=''):
        self.header = '[ISP]'
    
        self.ratio = cfgs.get('ratio', -1)
        self.anti_alias = cfgs.get('anti_alias', True)
        self.ext = cfgs.get('img_ext', 'jpg')
        self.do_ccm = cfgs.get('do_ccm', True)
        self.do_wb = cfgs.get('do_wb', True)
        self.do_gamma = cfgs.get('do_gamma', True)
        self.overwrite_name = cfgs.get('ovrw_name', False)

        self.find_static = cfgs.get('find_static', True)
        if self.find_static:
            self.img_buffers, self.img_diffs = [], []
            self.diff_min = float('inf')

        self.cfgs = cfgs
        
        if mdata_path != '':
            self.load_meta_data(mdata_path) # load meta data

        if img_dir != '' and img_list != '':
            self.config_inputs(img_dir, img_list)

    def load_meta_data(self, mdata_path):
        self.cam_data = np.load(mdata_path, allow_pickle=True)[()]
        for key in self.cam_data:
            self.prompt('%s: %s' % (key, self.cam_data[key]))

    def set_meta_data(self, mdata):
        self.cam_data = mdata
        for key in self.cam_data:
            self.prompt('%s: %s' % (key, self.cam_data[key]))

    def config_inputs(self, img_dir, img_list):
        self.prompt('Loading image list in %s' % (img_dir))
        self.img_paths, self.expos = self.load_img_list(img_dir, img_list)

    def load_img_list(self, img_dir='', img_list=''):
        filenames = []
        exposure_times = []
        img_expo_list = np.genfromtxt(os.path.join(img_dir, img_list), dtype='str')
        if img_expo_list.ndim == 1: # single image
            img_expo_list = img_expo_list.reshape(1, -1)
        img_paths = img_expo_list[:, 0]
        img_paths = [os.path.join(img_dir, path) for path in img_paths]
        expos = img_expo_list[:, 1].astype(np.float)
        return img_paths, expos
    
    def apply_white_balance(self, image, rgb_gains):
        # image RGB: [H, W, C], 
        r_gain, g_gain, b_gain = rgb_gains
        new_image = image.copy()
        new_image[:, :, 0] *= r_gain # red channel
        new_image[:, :, 2] *= b_gain # blue channel
        new_image = new_image.clip(0, 1)
        return new_image

    def demosaic(self, raw, cvt_type=cv2.COLOR_BAYER_RG2RGB):
        image = cv2.cvtColor(raw, cvt_type)
        #image = image.clip(0, 1)
        return image

    def apply_color_correction(self, image, ccm):
        # image RGB: [H, W, 3], ccm BGR [3, 3]
        image = image[:, :, np.newaxis, :]
        ccm = ccm[np.newaxis, np.newaxis, :, :]
        image = (image * ccm).sum(3)
        image = image.clip(0, 1)
        return image

    def apply_gamma(self, image, gamma=2.2):
        image = image.clip(1e-8, 1)
        image = np.power(image, 1.0 / gamma)
        return image
    
    def cvt_raw_to_rgb(self, save_dir, merge_hdr=False):
        img_paths = self.img_paths
        images = []
        utils.make_file(save_dir)  
        num_exps = len(np.unique(self.expos))
        
        for i, img_path in enumerate(img_paths):
            self.prompt('%d/%d: %s' % (i+1, len(img_paths), img_path))

            raw = np.load(os.path.join(img_path))

            rgb_image = self.camera_isp(raw)
        
            if self.find_static:
                self.check_img_abs_diff(rgb_image, num_exps)
        
            if merge_hdr:
                images.append(rgb_image)

            self.save_single_image(save_dir, img_path, rgb_image)

        if merge_hdr:
            self.merge_hdrs(save_dir, imgs=images, expos=self.expos)

        if self.find_static:
            diff_list_path = os.path.join(save_dir, 'img_diff_list.txt')
            self.save_img_diff_list(diff_list_path, img_paths[num_exps:])

    def save_single_image(self, save_dir, img_path, rgb_image):
        save_name = 'Img_%s_%.3fr.%s' % (os.path.basename(img_path)[:-4], self.ratio, self.ext)
        if self.anti_alias:
            save_name = save_name[:-4] + '_anti' + save_name[-4:]
        self.display(save_name)
        save_path = os.path.join(save_dir, save_name)
        if self.ext in ['jpg', 'png']:
            self.save_uint8(save_path, rgb_image)
        elif self.ext in ['tif']:
            self.save_uint16(save_path, rgb_image)

    def save_uint8(self, name, img):
        if img.dtype != np.uint8:
            img = (img.clip(0, 1) * 255).astype(np.uint8)
        imsave(name, img)

    def save_uint16(self, img_name, img):
        """img in [0, 1]"""
        img = img.clip(0, 1) * 65535
        img = img[:,:,[2,1,0]].astype(np.uint16)
        cv2.imwrite(img_name, img)

    def merge_hdrs(self, save_dir, imgs, expos):
        self.prompt('Merging HDRs for %s' % (save_dir))
        expos = expos / expos.min()
        if imgs is None:
            img_paths = self.img_paths
            imgs = []
            for i, img_path in enumerate(img_paths):
                raw = np.load(os.path.join(img_path))
                rgb_image = self.camera_isp(raw)
                imgs.append(rgb_image)

        hdr_merger = HDRMerger(bit=self.cfgs['bit'])
        hdr = hdr_merger.merge_hdrs(imgs, expos)
        utils.make_file(save_dir)  
        save_name = os.path.join(save_dir, '%s.hdr' % (os.path.basename(save_dir)))
        self.save_hdr(save_name, hdr)
        self.save_uint8(save_name[:-4] + '_loghdr.jpg', self.mulog_transform(hdr))

    def save_hdr(self, name, hdr):
        print(name)
        hdr = hdr[:, :, [2, 1, 0]].astype(np.float32)
        cv2.imwrite(name, hdr)

    def mulog_transform(self, in_tensor, mu=5000.0):
        denom = np.log(1.0 + mu)
        out_tensor = np.log(1.0 + mu * in_tensor) / denom 
        return out_tensor

    def camera_isp(self, raw):
        self.display('Demosaicing')
        demosaic_img = self.demosaic(raw)
        demosaic_img = demosaic_img[:,:,::-1] # BGR to RGB
        demosaic_img = demosaic_img.astype(np.double) / (2**self.cfgs['bit'] - 1)

        ratio = self.ratio
        if ratio > 0 and ratio < 1:
            h, w, c = demosaic_img.shape
            if self.anti_alias:
                kernel_size = self.get_smooth_kernel_size(factor=self.ratio)
                #print(kernel_size)
                demosaic_img = cv2.GaussianBlur(demosaic_img, kernel_size, sigmaX=0)
            demosaic_img  = cv2.resize(demosaic_img, (int(w*ratio), int(h*ratio)))

        if self.do_wb:
            self.display('White balance')
            wb_image = self.apply_white_balance(demosaic_img, self.cam_data['wb0'])
        else:
            wb_image = demosaic_img

        if self.do_ccm:
            self.display('Color correction')
            cc_image = self.apply_color_correction(wb_image, self.cam_data['ccm0'])
        else:
            cc_image = wb_image
        
        if self.do_gamma:
            self.display('Gamma correction')
            image = self.apply_gamma(cc_image)
        return image

    def get_smooth_kernel_size(self, factor):
        if factor == 0.5:
            return (3, 3)
        elif factor == 0.375:
            return (3, 3)
        elif factor in [0.2, 0.25]:
            return (5, 5)
        elif factor == 0.125:
            return (7, 7)
        else:
            raise Exception('Unknown factor')

    def display(self, string):
        mute = self.cfgs.get('mute', True)
        if not mute:
            print(string)

    def prompt(self, string):
        print('%s %s' % (self.header, string))

    def load_data(self, img_paths, bit=12):
        imgs = []
        for i, img_path in enumerate(img_paths):
            raw = np.load(os.path.join(img_dir, img_path))
            raw = raw.astype(np.double) / (2**bit - 1)
            imgs.append(raw)
        return imgs

    def check_img_abs_diff(self, cur_img, num_exps, thres=0.005):
        if len(self.img_buffers) == num_exps:
            diff = np.abs(cur_img - self.img_buffers[0]).sum() / cur_img.size
            self.prompt('Image Difference: %.8f' % diff)
            if diff < thres and diff < self.diff_min:
                self.prompt('*** Found static frames ***')
                self.diff_min = diff
            self.img_buffers.pop(0)
            self.img_diffs.append(diff)

        self.img_buffers.append(cur_img)

    def save_img_diff_list(self, list_path, img_paths):
        if len(img_paths) == 0:
            return
        img_diff_list = ['%s %.6f' % (os.path.basename(img), diff) for img, diff in zip(img_paths, self.img_diffs)]
        min_idx = np.argmin(self.img_diffs)
        img_diff_min = "Min: %s %.6f" % (os.path.basename(img_paths[min_idx]), self.img_diffs[min_idx])
        self.prompt(img_diff_min)
        img_diff_list.append(img_diff_min)
        utils.save_list(list_path, img_diff_list)


class HDRMerger(object):
    def __init__(self, bit=12):
        self.bit = bit # bit of LDRs

    def merge_hdrs(self, imgs, expos):
        # imgs: list of [H, W, 3], expos: n-vector
        imgs, expos = self.merge_same_exposure_imgs(imgs, expos)
        all_image = np.stack(imgs, 3) # [h, w, 3, N]
        ws = self.get_blend_weights(all_image) # [h, w, 3, N]

        fZ = np.power(all_image, 2.2) # linear radiance
        hdr = (ws * (fZ / expos[None, None, None])).sum(3) / (ws.sum(3) + 1e-8)
        return hdr

    def merge_same_exposure_imgs(self, imgs, expos):
        num_in = len(imgs)
        expo_img_dict = {}
        for expo, img in zip(expos, imgs):
            if expo not in expo_img_dict:
                expo_img_dict[expo] = []
            expo_img_dict[expo].append(img)

        new_imgs, new_expos = [], []
        for expo in expo_img_dict.keys():
            avg_img = np.stack(expo_img_dict[expo], 3).mean(3)
            new_imgs.append(avg_img)
            new_expos.append(expo)
            #cv2.imwrite('avg_img_%d.jpg' % expo, (avg_img[:,:,::-1] * 255).astype(np.uint8))

        print('[HDR] Merged same exposure images: %d->%d' % (num_in, len(new_imgs)))
        return new_imgs, np.array(new_expos).astype(np.float)

    def get_blend_weights(self, all_image, default=False):
        num_img = all_image.shape[3]
        if num_img > 3 or default == True: # SIGGRAPH 1997
            zmax = 2**self.bit - 1
            w = np.array([z if z < 0.5 * zmax else zmax - z for z in range(zmax+1)])
            Z = (all_image * zmax).astype(int) # scale to [0, 2**bit-1]
            ws = w[Z] # [h, w, 3, N]
            # Check if pixels over-exposed in all images
            if (all_image.sum(3) == num_img).sum() > 0: 
                oe_region = all_image.sum(3) == num_img # all 1
                ws[oe_region] = [1] + [0] * (num_img - 1) # [1, 0, 0, 0...]
        elif num_img == 3:
            ws = self.get_3exp_weights(all_image)
        elif num_img == 2:
            #ws = self.get_2exp_weights(all_image, ref_idx=1)
            ws = self.get_2exp_weights_sharp(all_image, ref_idx=1)
        else:
            raise Exception('Invalid image number: %d' % num_img)
        return ws

    def get_3exp_weights(self, x):
        low_exp_img, mid_exp_img, high_exp_img = np.split(x, 3, 3)
        low_index = (mid_exp_img <= 0.5).astype(float)
        low_exp_w = low_index * mid_exp_img * 2 + (1 - low_index) * 1
        mid_exp_w = low_index * mid_exp_img * 2 + (1 - low_index) * (1 - mid_exp_img) * 2
        high_exp_w = low_index * 1 + (1 - low_index) * (1 - mid_exp_img) * 2
        weight = np.concatenate([low_exp_w, mid_exp_w, high_exp_w], 3)
        return weight

    def get_2exp_weights(self, x, ref_idx=0):
        assert (ref_idx in [0, 1]) 
        ref_img = np.split(x, 2, 3)[ref_idx]
        low_index = (ref_img <= 0.5).astype(float)
        low_exp_w = low_index * ref_img * 2 + (1 - low_index) * 1
        high_exp_w = low_index * 1 + (1 - low_index) * (1 - ref_img) * 2
        weight = np.concatenate([low_exp_w, high_exp_w], 3)
        return weight

    def get_2exp_weights_sharp(self, x, ref_idx=0):
        assert (ref_idx in [0, 1])
        ref_img = np.split(x, 2, 3)[ref_idx]
        low_index = (ref_img <= 0.5).astype(float)
        low_left_part = 1 - np.sqrt((1 - (2*ref_img)**2).clip(0))
        low_exp_w = low_index * low_left_part + (1 - low_index) * 1
        
        high_right_part = 1 - np.sqrt((1 - (2*ref_img-2)**2).clip(0))
        high_exp_w = low_index * 1 + (1 - low_index) * high_right_part
        weight = np.concatenate([low_exp_w, high_exp_w], 3)
        return weight


def main(args):
    cfgs = vars(args)
    digital_isp = DigitalISP(cfgs, mdata_path=os.path.join(args.img_dir, args.cam_data),
            img_dir=args.img_dir, img_list=args.img_list)

    rgb_save_dir = os.path.join(args.img_dir)
    digital_isp.cvt_raw_to_rgb(utils.remove_slash(rgb_save_dir), merge_hdr=args.merge_hdr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--img_dir', default='.')
    parser.add_argument('--img_list', default='img_list.txt')
    parser.add_argument('--cam_data', default='cam_data.npy')
    parser.add_argument('--bit', default=12, type=int)
    parser.add_argument('--ratio', default=0.375, type=float)
    parser.add_argument('--img_ext', default='jpg')
    parser.add_argument('--ovrw_name', default=False, action='store_true') # overwrite
    parser.add_argument('--do_ccm', default=True, action='store_false')
    parser.add_argument('--do_wb', default=True, action='store_false')
    parser.add_argument('--do_gamma', default=True, action='store_false')
    parser.add_argument('--sharp_w', default=True, action='store_false')
    parser.add_argument('--anti_alias', default=True, action='store_false')
    parser.add_argument('--merge_hdr', default=False, action='store_true')
    parser.add_argument('--find_static', default=True, action='store_false')
    args = parser.parse_args()
    main(args)
