@echo off
rem python neural_style/neural_style.py train --net vgg16 --model-name wall-vgg16-1010-384.pth --image-size 384 --checkpoint-model-dir checkpoint/wall-384 --checkpoint-interval 33 --style-weight 1.0e10 --style-image style-images/red_brick_wall.jpg --save-model-dir models --epochs 1 --batch-size 2 > models\wall-vgg16-1010-384.log
rem python neural_style/neural_style.py train --model-name wall-vgg19-1010-512.pth --image-size 512 --style-weight 1.0e10 --style-image style-images/red_brick_wall.jpg --save-model-dir models --epochs 1 --batch-size 1 > models\wall-vgg19-1010-512.log
rem python neural_style/neural_style.py train --model-name wall-vgg19-1010-256.pth --image-size 256 --style-weight 1.0e10 --style-image style-images/red_brick_wall.jpg --save-model-dir models --epochs 1 --batch-size 1 > models\wall-vgg19-1010-256.log
rem python neural_style/neural_style.py train --net vgg19 --model-name wall_1024x576-vgg19-1010-256.pth --image-size 256 --style-weight 1.0e10 --style-image style-images/wall_1024x576.jpg --save-model-dir models --epochs 1 --batch-size 1 > models\wall_1024x576-vgg19-1010-256.log
rem python neural_style/neural_style.py train --net vgg16 --model-name wall_512x288-vgg16-1010-256.pth --image-size 256 --style-weight 1.0e10 --style-image style-images/wall_512x288.jpg --save-model-dir models --epochs 1 --batch-size 8 > models\wall_512x288-vgg16-1010-256.log

set argc=0
for %%x in (%*) do Set /A argc+=1

if not %argc% == 6 goto help
python neural_style/neural_style.py train --net vgg16 --model-name %2-%3-%4-%5.pth --image-size %5 --style-weight %1 --style-image style-images/%2.jpg --save-model-dir models --epochs 1 --batch-size %6 > logs\%2-%3-%4-%5.log

exit

:help
echo e.g. .\tr.cmd 1.0e10 image vgg16 1010 256 8
exit

rem .\tr.cmd 1.0e10 wall_768x432 vgg16 1010 256

