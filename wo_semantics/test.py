from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
#import tensorflow as tf
import tensorflow.compat.v1 as tf

import numpy as np
from PIL import Image
import math
import os
from six.moves import cPickle as pickle
from six.moves import range

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

import deploy
FLAGS = tf.app.flags.FLAGS

MODEL_NAME = "model.ckpt"
image_size = 256
n=7404
model_num='246480' #指定模型id
#compathfile ='/mnt/liguanlin/DataSets/iHarmony4/' + 'IHD_noisy25_test.txt' #在整个数据集上预测
#compathfile ='/mnt/liguanlin/DataSets/iHarmony4/Hday2night/' + 'Hday2night_test.txt' #在Hday2night数据集上预测
#compathfile ='/mnt/liguanlin/DataSets/iHarmony4/HFlickr/' + 'HFlickr_test.txt'
#compathfile ='/mnt/liguanlin/DataSets/iHarmony4/HCOCO/' + 'HCOCO_test.txt'
compathfile ='/mnt/liguanlin/DataSets/iHarmony4/HAdobe5k/' + 'HAdobe5k_test.txt'

file = open(model_num+'.txt', 'a')

def _get_mask_path_(path):
  split_slash=path.split("/")
  split_underline=split_slash[-1].split("_")
  mask_path=split_slash[0]+"/masks/"+split_underline[0]+"_"+split_underline[1]+".png"
  return mask_path

def _get_truth_path_(path):
  split_slash=path.split("/")
  split_underline=split_slash[-1].split("_")
  truth_path=split_slash[0]+"/real_images/"+split_underline[0]+".jpg"
  return truth_path


def get_name(path):
  slash = path.split("/")
  return slash[-1]

def _parse_image(image_path):
  im_ori = Image.open(image_path)
  im_ori = im_ori.convert('RGB')
  im = im_ori.resize(np.array([image_size,image_size]), Image.BICUBIC)
  im = np.array(im, dtype=np.float32)
  if im.shape[2] == 4:
    im = im[:,:,0:3]
  im = im[:,:,::-1]
  im -= np.array((127.5, 127.5, 127.5))
  im = np.divide(im, np.array(127.5))
  image = im[np.newaxis, ...]
  return image

def _parse_mask(mask_path):
  mask = Image.open(mask_path)
  mask = mask.resize(np.array([image_size,image_size]), Image.BICUBIC)
  mask = np.array(mask, dtype=np.float32)
  if len(mask.shape) == 3:
    mask = mask[:,:,0]
  mask -= 127.5
  mask = np.divide(mask, np.array(127.5))
  mask = mask[np.newaxis, ...]
  mask = mask[..., np.newaxis]
  return mask


def _parse_truth_eval(truth_path):
  truth = Image.open(truth_path)
  truth = truth.convert('RGB')
  truth = truth.resize(np.array([image_size,image_size]), Image.BICUBIC)
  truth = np.array(truth, dtype=np.float32)
  if truth.ndim == 2:
    truth = truth.reshape(np.array((512,512,3)))
  return truth


path_img = []
path_mask = []
path_truth = []
with open(compathfile,'r') as f:
  for line in f.readlines():
    #path_img.append(line.rstrip())
    #path_mask.append(_get_mask_path_(line.rstrip()))
    #path_truth.append(_get_truth_path_(line.rstrip()))

    #以下代码用来在各个子数据集上进行预测
    path_img.append('composite_noisy25_images/' + line.rstrip())

    name_parts=line.split('_')
    mask_path = line.replace(('_'+name_parts[-1]),'.png')
    path_mask.append('masks/' + mask_path)
    gt_path = line.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
    path_truth.append('real_images/'+ gt_path)



tf.disable_eager_execution()
com_placeholder = tf.placeholder(tf.float32,
                                shape=(1, image_size, image_size, 3))
masks_placeholder = tf.placeholder(tf.float32,
                                shape=(1, image_size, image_size, 1))

harmnization = deploy.inference(com_placeholder, masks_placeholder)

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
new_saver = tf.train.import_meta_graph('model/model.ckpt-'+model_num+'.meta')
new_saver.restore(sess, 'model/model.ckpt-'+model_num)
if new_saver:
  print("Restore successfully!")

sample_num = len(path_img)
mse_score_sum = 0
psnr_score_sum = 0
ssim_score_sum = 0

for i in range(0,len(path_img)):
  com = _parse_image(os.path.join(FLAGS.data_dir,path_img[i]))
  mask = _parse_mask(os.path.join(FLAGS.data_dir,path_mask[i]))
  truth = _parse_truth_eval(os.path.join(FLAGS.data_dir,path_truth[i]))
  feed_dict = {
  com_placeholder: com,
  masks_placeholder: mask,
  }
  harm = sess.run(harmnization, feed_dict=feed_dict)
  harm_rgb=np.squeeze(harm)
  harm_rgb=np.multiply(harm_rgb,np.array(127.5))
  harm_rgb += np.array((127.5, 127.5, 127.5))
  harm_rgb = harm_rgb[:,:,::-1]
  neg_idx = harm_rgb < 0.0
  harm_rgb[neg_idx] = 0.0
  pos_idx = harm_rgb > 255.0
  harm_rgb[pos_idx] = 255.0
  name=get_name(path_img[i])

  truth1 = Image.open(os.path.join(FLAGS.data_dir,path_truth[i]))
  truth1 = truth1.resize([256,256], Image.BICUBIC)
  truth1 = np.array(truth1, dtype=np.float32)

  mse_score = mse(harm_rgb,truth1)
  psnr_score = psnr(truth1,harm_rgb,data_range=harm_rgb.max() - harm_rgb.min())
  ssim_score = ssim(truth1,harm_rgb,data_range=harm_rgb.max() - harm_rgb.min(),multichannel=True)
  
  file.writelines('%s\t%f\t%f\t%f\n' % (name,mse_score,psnr_score,ssim_score))
  print(i,name,mse_score,psnr_score,ssim_score)

  mse_score_sum += mse_score
  psnr_score_sum += psnr_score
  ssim_score_sum += ssim_score

  """下面分别计算fmse, fpsnr和fssim"""
  """
  mask = Image.open(os.path.join(FLAGS.data_dir,path_mask[i])).convert('1') #获取mask区域。
  mask = tf.resize(mask, [image_size,image_size], interpolation=Image.BICUBIC)

  mask = tf.to_tensor(mask).unsqueeze(0).cuda()
  harmonized = tf.to_tensor(harmonized_np).unsqueeze(0).cuda()
  real = tf.to_tensor(real_np).unsqueeze(0).cuda()

  fore_area = torch.sum(mask)
  fmse_score = torch.nn.functional.mse_loss(harmonized*mask,real*mask)*256*256/fore_area #计算得到fmse        
  fmse_score = fmse_score.item()

  fpsnr_score = 10 * np.log10((255 ** 2) / fmse_score) #计算得到fpsnr
  
  ssim_score, fssim_score = pytorch_ssim.ssim(harmonized, real, window_size=ssim_window_default_size, mask=mask) #计算得到fssim
  fmse_scores += fmse_score
  fpsnr_scores += fpsnr_score
  fssim_scores += fssim_score
  """
mse_nu = mse_score_sum/sample_num
psnr_mu = psnr_score_sum/sample_num
ssim_mu = ssim_score_sum/sample_num

print("mean=", mse_nu, psnr_mu, ssim_mu)

sess.close()
print('Done!')