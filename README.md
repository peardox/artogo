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

##### Training Data (optional)

Training data may be obtained however you like but I have prepared a specially packaged variant of Coco2017 that you can download from https://peardox.com/downloads/train.tar.gz (4.7Gb - jpegs don't compress so extracted it becomes the same size roughly)

The file layout of this download is as follows

train
\+ coco100
\+ coco2017
     \+ 512
             \+01.. 17
                  \+ 01 .. 18
                       \+ 01 .. 18

The coco2017/512 directory therefore contains up to 19 directories three levels deep with files named according to their directory and sequence in that folder - e.g. coco2017/512/01/02/03/01-02-03-01.jpg to 01-02-03-18.jpg. While this may look stupid it's far from it. All the files are pre-processed to be 512x512 in size (saves a resize in code) and the layout ensures files can be accessed quickly. The original release has files of wildly varying image dimensions and places all of them in one directory (118,287 files in one directory is a bad idea...)

If used the archive should be expanded in the same directory as the root of this repo as it will extract into the train directory (or you could end up with train/train)

The point of this training set is proper layout and sizing for the project - the unmodified coco2017 is much larger at 18Gb

