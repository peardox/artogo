@echo off
rem .\movie.cmd haywain mosaic vgg16 1010 ckpt

set argc=0
for %%x in (%*) do Set /A argc+=1

if %argc% lss 5 goto help

echo python neural_style/neural_style.py eval --model-dir models --model  %2-%3-%4-%5 --content-image input-images/%1.jpg --output-image output-images/%1-%2-%3-%4-%5.jpg --movie 1
python neural_style/neural_style.py eval --model-dir models --model %2-%3-%4-%5 --content-image input-images/%1.jpg --output-image output-images/%1-%2-%3-%4-%5.jpg --movie 1
exit

:help
echo e.g. .\st.cmd image model vgg16 1010 256
exit
