--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'
require 'ffmpeg'

local M = {}
local UCF101Dataset = torch.class('resnet.UCF101Dataset', M)

function UCF101Dataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir   = opt.data
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function UCF101Dataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      target = class,
   }
end

function UCF101Dataset:_loadImage(path)
   local ok, input = pcall(function()
      flowpath = string.gsub(path, '.avi$', '_flow.avi')
      local testvidname = flowpath
      videoinput = ffmpeg.Video{path=testvidname, width=224, height=224, 
                         fps=30, length=2, delete=false, 
                         destFolder='.ffmpeg-tmp-ucf-101',silent=true}

      local vidtensor = videoinput:totensor{}
      local vidframes = vidtensor:size(1)
      return vidtensor[30]
      --return image.load(path, 3, 'float')
   end)
   maxvidframes = 57
   --print ('Loading path'..path)
   flowpath = string.gsub(path, '.avi', '_flow.avi')
   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local testvidname = flowpath
      --print ('Flowpath'..flowpath)
      videoinput = ffmpeg.Video{path=testvidname, width=224, height=224, 
                         fps=30, length=2, delete=false, 
                         destFolder='.ffmpeg-tmp-ucf101',silent=true}

      local vidtensor = videoinput:totensor{}
      local vidframes = vidtensor:size(1)
      return vidtensor[10]
   end

   return input
end

function UCF101Dataset:size()
   print ('size is '..self.imageInfo.imageClass:size(1))
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function UCF101Dataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.RandomSizedCrop(224),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.UCF101Dataset
