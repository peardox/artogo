import argparse
import os
import sys
import time
import logging
import torch
from mlfuncts import *
from cmdfuncts import *


def check_gpu():
    try:
        torch.cuda.init()
        if(torch.cuda.is_available()):
            gpu_supported = 1
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
        gpu_supported = 0

    print("GPU Support : ", gpu_supported);
    return gpu_supported

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
    train_arg_parser.add_argument("--force-size", type=int, default=1,
                                  help="If set to 1 all training images are resized")
    train_arg_parser.add_argument("--dataset", type=str, default="train/coco2017/512",
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--model-name", type=str, default="test",
                                  help="model name")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=384,
                                  help="size of training images, default is 384 X 384")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--logfile", type=str, default=None,
                                  help="Optional lof file location")
    train_arg_parser.add_argument("--ignore-gpu", type=int, default=0,
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
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=1000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--model-dir", type=str, default="models",
                                 help="Path to saved models")
    eval_arg_parser.add_argument("--ignore-gpu", type=int, default=0,
                                  help="Set it to 1 to ignore GPU if detected")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")
    eval_arg_parser.add_argument("--movie", type=str, default=None,
                                 help="path to movie styles")
    eval_arg_parser.add_argument("--add-model-path", type=int, default=1,
                                 help="Add movie path ot not")
    eval_arg_parser.add_argument("--logfile", type=str, default=None,
                                  help="Optional lof file location")
    args = main_arg_parser.parse_args()
    
    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
        
    use_gpu = 0
    if not args.ignore_gpu:
        use_gpu = check_gpu()

    if args.subcommand == "train" and args.logfile is not None:
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
                if use_gpu:
                    torch.cuda.empty_cache()
                if trial_batch == 0:
                    print("No batch size found to run current training session (style image too large)")
                    sys.exit(1)
    else:
        if args.movie is None:
            stylize(args, use_gpu)
        else:
            frame_id = 0;
            main_model = args.model
            # assign directory
            directory = os.path.join('movies', main_model)
             
            # iterate over files in
            # that directory
            for filename in os.listdir(os.path.join(args.model_dir, directory)):
                f = os.path.join(args.model_dir, directory, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    # print(f)
                    frame_id += 1;
                    args.model = os.path.join(directory, filename)
                    args.output_image = os.path.join('movies', main_model, str(frame_id).zfill(4)) + '.jpg'
                    args.add_model_path = 0;
                    print(args.model, " -> ", args.output_image)
            # print("looping : ", x)
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
