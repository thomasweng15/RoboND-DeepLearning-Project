import os
import glob
from scipy import misc
import numpy as np

def flip_and_save_images(img_dir, extension):
  os.chdir(img_dir)
  files = glob.glob("*." + extension)
  for i, file in enumerate(files):
    print(i)
    img = misc.imread(file, flatten=False, mode='RGB')
    flipped_img = np.fliplr(img)
    misc.imsave("flipped" + file, flipped_img)

flip_and_save_images("../data/train/images", "jpeg")
flip_and_save_images("../masks", "png")

flip_and_save_images("../../validation/masks", "png")
flip_and_save_images("../images", "jpeg")