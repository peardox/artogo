import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
# import vgg
from vgg import *
# from vgg import Vgg16
import logging
import pynvml

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
                print("CUDA Device Memory : ",torch.cuda.mem_get_info(x))
                print("CUDA Device Properties : ",torch.cuda.get_device_properties(x))
                # print(torch.cuda.memory_summary(x))
    except:
        print("No supported GPUs detected")
        gpu_supported = 0

    print("GPU Support : ", gpu_supported);
    return gpu_supported
    

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args, use_gpu):
    device = torch.device("cuda" if use_gpu else "cpu")
    if args.limit != 0:
        limit = args.limit
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.force_size == 1:
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    if args.net.casefold() == 'vgg16':
        vgg = Vgg16(requires_grad=False).to(device)
    else:
        vgg = Vgg19(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

#    sbtrcnt = 0
    
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        ckpt_id = 0
        
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

#            sbtrcnt += 1
#            print("sb = ", sbtrcnt, " - n_batch = ", n_batch, " - count = ", count, " - usc = ", _, "Size = ", sys.getsizeof(x))
            
            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if ((batch_id + 1) % args.log_interval == 0) or (batch_id == 0 and e == 0):
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                logging.info(mesg)
                print("\r" + mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_" + str(ckpt_id + 1).zfill(4) + ".pth"
                ckpt_id += 1;
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()
                
            if args.limit != 0 and count >= limit:
                break;

    # save model
    transformer.eval().cpu()
    # save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_filename = args.model_name + '.pth'
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
        time.ctime(), e + 1, count, len(train_dataset),
                      agg_content_loss / (batch_id + 1),
                      agg_style_loss / (batch_id + 1),
                      (agg_content_loss + agg_style_loss) / (batch_id + 1)
    )
    logging.info(mesg)
    print("\r" + mesg + "\n")
    print("\nDone, trained model saved at", save_model_path)
#    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

def stylize(args, use_gpu):
    device = torch.device("cuda" if use_gpu else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            if args.add_model_path == 1:
                state_dict = torch.load(os.path.join(args.model_dir, args.model + '.pth'))
            else:
                state_dict = torch.load(os.path.join(args.model_dir, args.model))
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()            
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])
    # if use_gpu:
    #     print(torch.cuda.memory_summary(0))

def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)


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
    
    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile, encoding='utf-8', format='%(message)s', level=logging.INFO)
        print("Logging to ", args.logfile)

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
        
    use_gpu = 0
    if args.ignore_gpu == 0:
        use_gpu = check_gpu()

    if args.subcommand == "train":
        check_paths(args)
        train(args, use_gpu)
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
