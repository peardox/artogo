echo python neural_style/neural_style.py train --net vgg16 --model-name %2-%3-%4-%5 --image-size %5 --style-weight %1 --style-image style-images/%2.jpg --save-model-dir models --epochs 5 --logfile logs\%2-%3-%4-%5.log
python neural_style/neural_style.py train --net vgg16 --model-name %2-%3-%4-%5 --image-size %5 --style-weight %1 --style-image style-images/%2.jpg --save-model-dir models --epochs 1 --batch-size %6 --logfile logs\%2-%3-%4-%5.log %7 %8

