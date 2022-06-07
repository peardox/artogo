@echo off
set argc=0
for %%x in (%*) do Set /A argc+=1

if %argc% lss 5 goto help

echo python pysrc/fns.py eval --model-dir models --model  %2-%3-%4-%5 --content-image input-images/%1.jpg --output-image output-images/%1-%2-%3-%4-%5.jpg
python pysrc/fns.py eval --model-dir models --model %2-%3-%4-%5 --content-image input-images/%1.jpg --output-image output-images/%1-%2-%3-%4-%5.jpg
exit

:help
echo e.g. .\st.cmd image model vgg16 1010 256
exit
