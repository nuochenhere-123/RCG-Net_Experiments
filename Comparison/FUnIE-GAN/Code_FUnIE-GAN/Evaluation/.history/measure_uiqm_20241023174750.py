"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from uqim_utils import getUIQM


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

def measure_UIQMs1(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*out.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
#inp_dir = "/home/xahid/datasets/released/EUVP/test_samples/Inp/"
inp_dir = "./New4_4_C1-U45"
## UIQMs of the distorted input images 
inp_uqims = measure_UIQMs1(inp_dir)
print ("Our UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))

## UIQMs of the enhanceded output images
#gen_dir = "eval_data/euvp_test/funie-gan/" 
gen_dir = "./UShape-U45" 
gen_uqims = measure_UIQMs(gen_dir)
print ("UShape Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

gen_dir = "./UColor-U45" 
gen_uqims = measure_UIQMs(gen_dir)
print ("Ucolor Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

gen_dir = "./WaterNet-ReTrain-U45" 
gen_uqims = measure_UIQMs(gen_dir)
print ("WaterNet Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))



