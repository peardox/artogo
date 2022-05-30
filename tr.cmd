python neural_style/neural_style.py train --net vgg16 --model-name $2-$3-$4-$5.pth --image-size $5 --style-weight $1 --style-image style-images/$2.jpg --save-model-dir models --epochs 1 --batch-size $6 > logs\$2-$3-$4-$5.log
echo e.g. ./tr 1.0e10 wall_768x432 vgg16 1010 256

