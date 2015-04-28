#!/bin/bash

# #SBATCH -o jjl5sw.test.out.2
#SBATCH -p qdata


echo "Starting in the directory: `pwd`"
cd $HOME/GitHub/Scene-Labeling-Conv-Net/
echo "changed to the directory: `pwd`"


# nhu='25,50'
# pools='8,2'
# conv_kernels='6,3,7'
# nhu='16,32,64'
# pools='2,2,2'
# conv_kernels='4,3,3,7'
nhu='64,64,64,64'
pools='2,2,2,2'
conv_kernels='6,5,5,5,4'

dropout='0.5'
indropout='0.2'
num_train_imgs='572'


/usr/cs/bin/th main.lua -nhu $nhu -pools $pools -conv_kernels $conv_kernels -num_train_imgs $num_train_imgs -dropout $dropout -indropout $indropout #-create_shifted_inputs $create_shifted_inputs


