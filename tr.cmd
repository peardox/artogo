@echo off
set argc=0
for %%x in (%*) do Set /A argc+=1

if %argc% lss 5 goto help

echo python pysrc/fns.py train --net %3 --model-name %2-%3-%4-%5 --image-size %5 --style-weight %1 --style-image style-images/%2.jpg --save-model-dir models --epochs 5 --logfile logs\%2-%3-%4-%5.log
python pysrc/fns.py train --net %3 --model-name %2-%3-%4-%5 --image-size %5 --style-weight %1 --style-image style-images/%2.jpg --save-model-dir models --epochs 1 --batch-size %6 --logfile logs\%2-%3-%4-%5.log %7 %8

exit

:help
echo tr.cnd weight style net rev-weight image-size batch-size 
echo e.g. .\tr.cmd 1.0e10 brick vgg16 1010 256 4
exit
