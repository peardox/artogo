# artogo
Fast Neural Style Adaption

Based on https://github.com/pytorch/examples/tree/main/fast_neural_style

##### Important new switch

--model-name (for Windows compatibility)

##### Sample Usage

python neural_style/neural_style_fixed.py train --model-name hero-vgg16-1010-512.pth --image-size 512 --style-weight 1.0e10 --style-image style-images/hero.jpg --save-model-dir models --epochs 1 --batch-size 2

##### Model Naming

This is only a suggestion but naming models with the base net (e.g. vgg16), style-image reverse padded (1010) and image size used to create (512) is how I'm coding them for legibility ATM

##### Reversed Weight naming

Again a suggestion - I name weights in filenames to make them easy to sort so 1.0e10 becomes 1010 - this is exponent followed by 2 digit mantissa (which 1010 doesn't explain at all) - a proper example is that 7.5e10 would become 1075. The reason for this naming is that bigger numbers when sorted ascending appear lower down in the list (adapted from a similar convention where dates are written YYYYMMDD)

