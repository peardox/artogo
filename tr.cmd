@echo off
set argc=0
for %%x in (%*) do Set /A argc+=1

if %argc% lss 5 goto help

echo python pysrc/fns.py train --net %3 --model-name %2-unsplash-%7e-%3-%4-%5 --image-size %5 --style-weight %1 --style-image style-images/%2.jpg --save-model-dir models --epochs %7 --batch-size %6 --logfile logs\%2-unsplash-%7e-%3-%4-%5.log %8 %9
python pysrc/fns.py train --net %3 --model-name %2-unsplash-%7e-%3-%4-%5 --image-size %5 --style-weight %1 --style-image style-images/%2.jpg --save-model-dir models --epochs %7 --batch-size %6 --logfile logs\%2-unsplash-%7e-%3-%4-%5.log %8 %9 

exit

:help
echo tr.cnd weight style net rev-weight image-size batch-size epochs
echo e.g. .\tr.cmd 1.0e10 brick vgg16 1010 256 4 1
exit

