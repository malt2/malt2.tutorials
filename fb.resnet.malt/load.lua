--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
local DataLoader = require 'dataloader'
local opts = require 'opts'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)


-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

print(string.format(' * Finished data download. Comment this script after first use in mpitest.sh'))
