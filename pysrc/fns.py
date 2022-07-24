import argparse
import os
import sys
import time
import logging
import torch
from mlfuncts import *

def check_gpu():
    try:
        torch.cuda.init()
        if(torch.cuda.is_available()):
            gpu_supported = True
            print("CUDA Available : ",torch.cuda.is_available())
            print("CUDA Devices : ",torch.cuda.device_count())
            print("CUDA Arch List : ",torch.cuda.get_arch_list())
            for x in range(torch.cuda.device_count()):
                print("CUDA Capabilities : ",torch.cuda.get_device_capability(x))
                print("CUDA Device Name : ",torch.cuda.get_device_name(x))
                # logging.info("CUDA Device Name : " + torch.cuda.get_device_name(x))
                print("CUDA Device Memory : ",torch.cuda.mem_get_info(x))
                print("CUDA Device Properties : ",torch.cuda.get_device_properties(x))
                # print(torch.cuda.memory_summary(x))
    except:
        print("No supported GPUs detected")
        gpu_supported = False

    print("GPU Support : ", gpu_supported);
    return gpu_supported

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def main():
    start = time.time()
    
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 1")
    train_arg_parser.add_argument("--limit", type=int, default=0,
                                  help="Limit training data to this value, default is 0 (no limit)")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--force-size", type=str2bool, default=True,
                                  help="If set to 1 all training images are resized")
    train_arg_parser.add_argument("--dataset", type=str, default="train/coco2017/512",
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--model-name", type=str, default="test",
                                  help="model name")
    train_arg_parser.add_argument("--model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default="",
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--model-ext", type=str, default=".pth",
                                  help="model extension (include dot)")
    train_arg_parser.add_argument("--style-scale", type=float, default=1,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--channels", type=int, default=32,
                                  help="Channels to use in training")
    train_arg_parser.add_argument("--logfile", type=str, default="",
                                  help="Optional lof file location")
    train_arg_parser.add_argument("--ignore-gpu", type=str2bool, default=False,
                                  help="Set it to 1 to ignore GPU if detected")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--net", type=str, default="vgg19",
                                  help="Network to use - vgg19 or vgg16")
    train_arg_parser.add_argument("--log-event-api", type=str2bool, default=False,
                                  help="Only used by Delphi")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-image-raw", type=str, default="",
                                 help="Raw content image (Placeholder)")
    eval_arg_parser.add_argument("--content-scale", type=float, default=1,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--model-dir", type=str, required=True,
                                 help="Path to saved models")
    eval_arg_parser.add_argument("--model-ext", type=str, default=".pth",
                                  help="model extension (include dot)")
    eval_arg_parser.add_argument("--log-event-api", type=str2bool, default=False,
                                  help="Only used by Delphi")
    eval_arg_parser.add_argument("--ignore-gpu", type=str2bool, default=False,
                                  help="Set it to 1 to ignore GPU if detected")
    eval_arg_parser.add_argument("--export_onnx", type=str2bool, default=False,
                                 help="export ONNX model to a given file")
    eval_arg_parser.add_argument("--add-model-ext", type=str2bool, default=True,
                                 help="Add model ext or not")
    eval_arg_parser.add_argument("--logfile", type=str, default="",
                                  help="Optional log file location")
    args = main_arg_parser.parse_args()
    
    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
        
    use_gpu = False
    if not args.ignore_gpu:
        use_gpu = check_gpu()

    if args.subcommand == "train" and (args.logfile != ""):
        logging.basicConfig(filename=args.logfile, encoding='utf-8', format='%(message)s', level=logging.INFO, filemode='w')
        print("Logging to ", args.logfile)

    if args.subcommand == "train":
        check_paths(args)
        trial_batch = args.batch_size
        
        while(1):
            oom = False
            try:
                print("Trying batch of ", trial_batch)
                train(args, use_gpu, trial_batch)
            except RuntimeError as e:
                print("Hit exception handler")
                if trial_batch > 0:
                    oom = True
                else:
                    print(e)
                    sys.exit(1)
            else:
                break

            if oom:
                trial_batch -= 1
                if trial_batch == 0:
                    print("No batch size found to run current training session (style image too large)")
                    sys.exit(1)
    else:
        stylize(args, use_gpu)
        
    elapsed = time.time() - start    
    print("Elapsed time = %f secs" % (elapsed))
    hour = elapsed // 3600
    elapsed %= 3600
    minutes = elapsed // 60
    elapsed %= 60
    seconds = elapsed
    print("Elapsed time = %d hours %d mins %d secs" % (hour, minutes, seconds))

if __name__ == "__main__":
    main()

'''
# python pysrc/fns.py train --epochs 1 --batch-size 1 --dataset /train/unsplash/256 --style-image style-images/flowers.jpg --model-dir models --model-name flowers-256 --style-weight 1e10 --net vgg16 --logfile logs/flowers-256.csv
# python pysrc/fns.py eval --content-image input-images/haywain.jpg --model-dir models --model flowers-256 --output-image output-images/fns-test-flowers.png
# python pysrc/fns.py train --epochs 4 --batch-size 10 --dataset /train/unsplash/256 --style-image style-images/flowers.jpg --model-dir models --model-name flowers-256-4 --style-weight 1e10 --net vgg16 --logfile logs/flowers-256-4
# python pysrc/fns.py eval --content-image input-images/haywain.jpg --model-dir models --model flowers-256-4 --output-image output-images/fns-test-flowers-4.png
# python pysrc/fns.py train --epochs 2 --batch-size 2 --dataset /train/unsplash/256 --style-image style-images/operagx.jpg --model-dir models --model-name operagx-256-2 --style-weight 1e10 --net vgg16 --logfile logs/operagx-256-2.csv
# python pysrc/fns.py eval --content-image input-images/haywain-wall.jpg --model-dir models --model operagx-256-2 --output-image output-images/logocomp.png
# python pysrc/fns.py train --epochs 4 --batch-size 20 --dataset /train/unsplash/256 --style-image style-images/stray.jpg --model-dir models --model-name stray-256-4 --style-weight 1e10 --net vgg19 --logfile logs/stray-256-4.csv --log-event-api True
# python pysrc/fns.py train --epochs 16 --batch-size 20 --dataset /train/unsplash/256 --style-image style-images/stray.jpg --model-dir models --model-name stray-256-16 --style-weight 1e10 --net vgg19 --logfile logs/stray-256-16.csv --checkpoint-model-dir checkpoints --log-event-api True
# python pysrc/fns.py eval --model-dir models --model wall --content-image input-images/haywain.jpg --output-image output-images/haywain-wall.jpg

# train --epochs 1 --batch-size 1 --dataset /git/artogo/datasets/train/unsplash/lite/256 --style-image /dae/dae/256/dae_mosaic_1.jpg --model-dir models --model-name test-time --style-weight 1e10 --net vgg16 --log-event-api True

# eval --model-dir models --model dae_mosaic_1-256 --content-image input-images/haywain.jpg --output-image output-images/aaa-haywain-mosaic-256.jpg

train --epochs 1 --batch-size 1 --dataset datasets/train/unsplash/lite/256 --style-image style-images/dae_mosaic_1.jpg --model-dir models --model-name test-time --style-weight 1e10 --net vgg16 --log-event-api True


'''

