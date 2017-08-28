# Download the dataset first
`which th` `pwd`/load.lua -dataset cifar10
mpirun -np 2 ./redirect.sh `which th` `pwd`/main.lua -dataset cifar10 -batchSize 128 -depth 110
