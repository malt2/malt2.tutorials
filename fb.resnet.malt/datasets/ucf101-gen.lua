--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}


   numclasses  = 101
  -- classes in UCF-101 -- This is needed because the test files do not have class labels in ucf-101
  classList = {"ApplyLipstick", "BasketballDunk","BoxingSpeedBag", "BreastStroke",  "FloorGymnastics", "IceDancing", "Lunges", "SkyDiving", "MilitaryParade", 
      "HulaHoop", "VolleyballSpiking", "Skijet", "JavelinThrow", "LongJump", "Mixing", "Shotput", "BandMarching", "Kayaking", "StillRings", "PlayingPiano",
      "PushUps", "Archery", "FieldHockeyPenalty", "BoxingPunchingBag", "PlayingCello", "FrontCrawl", "Billiards", "Rowing",  "TrampolineJumping",  "Punch",
      "CuttingInKitchen", "BodyWeightSquats", "JugglingBalls", "Nunchucks", "JumpRope", "PlayingViolin", "PlayingGuitar", "YoYo", "SumoWrestling", "SoccerJuggling",
      "CliffDiving", "CricketBowling", "PlayingDhol", "HorseRiding", "BabyCrawling", "PlayingSitar", "TaiChi", "BenchPress", "PommelHorse", "BrushingTeeth",
      "Hammering", "PlayingTabla", "HandstandWalking", "Typing", "CleanAndJerk", "TennisSwing", "CricketShot", "BlowDryHair", "HeadMassage", "BalanceBeam",
      "TableTennisShot", "MoppingFloor", "Drumming", "PlayingFlute", "FrisbeeCatch", "ApplyEyeMakeup", "SkateBoarding", "BaseballPitch", "SoccerPenalty", "ThrowDiscus",
      "RopeClimbing", "HorseRace", "HighJump", "PullUps", "Diving",  "ParallelBars", "WalkingWithDog", "PizzaTossing", "BlowingCandles",  "Swing",
      "GolfSwing", "PoleVault", "UnevenBars", "HandstandPushups", "JumpingJack", "WallPushups", "WritingOnBoard", "Skiing", "Bowling", "Surfing",
      "SalsaSpin", "ShavingBeard", "Basketball", "Knitting", "RockClimbingIndoor", "Haircut", "Biking", "Fencing", "Rafting", "PlayingDaf",
      "HammerThrow"
  }
  table.sort(classList)
  
  for l = 1,#classList do
    class = classList[l]
    classToIdx[class] = l
    --print (traininglist)
  end

   -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
   return classList, classToIdx
end

local function findImages(dir, classToIdx, videolist)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   totalvideos = table.getn(videolist)
   imagePath = videolist

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)
      local path = className .. '/' .. filename

      local classId = classToIdx[className]
      assert(classId, 'class not found: ' .. className)

      table.insert(imagePaths, path)
      table.insert(imageClasses, classId)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   local imageClass = torch.LongTensor(imageClasses)
   return imagePath, imageClass
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = opt.data
   local valDir = opt.data
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(trainDir)

   local splitfiles = {train = './ucfTrainTestlist/trainlist0'..opt.ucf101split..'.txt', test = './ucfTrainTestlist/testlist0'..opt.ucf101split..'.txt'}
  -- split files have video names and class names
   traininglist = {}
   testlist = {}
   trimagePaths = {}
   trimageClasses = {}
   local maxLength = -1
   for l in io.lines(splitfiles.train) do
     
     local f, c = l:match '(%S+)%s+(%S+)'
     table.insert(traininglist, { filename = f, class = c})
     table.insert(trimagePaths, f)
     table.insert(trimageClasses, c)
     --print (traininglist)
     maxLength = math.max(maxLength, #f + 1)
   end
   local nImages = #trimagePaths
   trainImagePath = torch.CharTensor(nImages, maxLength):zero()
   print ('Total TR images:'.. table.getn(trimagePaths))
   for i, path in ipairs(trimagePaths) do
      ffi.copy(trainImagePath[i]:data(), path)
   end

   trainImageClass = torch.LongTensor(trimageClasses)

     -- print (table.getn(traininglist))
     -- print (traininglist)
     -- print (traininglist[1].filename)
   totaltrvideos = table.getn(traininglist)

   vaimagePaths = {}
   vaimageClasses = {}
   for m in io.lines(splitfiles.test) do
     
     local f = m:match '(%S+)'
     local cstr, fstr = m:match '(%a+)/(%a+)'
     c = classToIdx[cstr]
     --print (cstr, c)
     table.insert(testlist, {filename = f, class = c})
     table.insert(vaimagePaths, f)
     table.insert(vaimageClasses, c)
     maxLength = math.max(maxLength, #f + 1)
    --print (testlist)
   end
   local nImages = #vaimagePaths
   print ('Total VA images'..nImages)
   valImagePath = torch.CharTensor(nImages, maxLength):zero()

   for i, path in ipairs(vaimagePaths) do
      print ('path is '.. path)
      ffi.copy(valImagePath[i]:data(), path)
   end

   valImageClass = torch.LongTensor(vaimageClasses)

   totaltestvideos = table.getn(testlist)

   print(" | finding all validation images")
   --valImagePath = testlist
   --valImageClass =  classList

   print(" | finding all training images")
   --trainImagePath = traininglist
   --trainImageClass = classList

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
