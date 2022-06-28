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
from vgg import *
import logging
from cmdfuncts import *

def show_gpu_memory(label):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(label, " => Free : ", f, "Reserved : ", r, "Allocated : ", a, "Total : ", t)

def check_paths(args):
    try:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        if (args.checkpoint_model_dir != "") and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)
        

def train(args, use_gpu, trial_batch_size):
    abort_flag = False
    except_flag = False
    e = 0
    agg_content_loss = 0
    agg_style_loss = 0.
    count = 0
    image_count = 0
    ckpt_id = 0
    batch_id = 0
    reporting_interval = 1
    train_start = time.time()
    train_reported = 0
    reporting_line = 0
    total_images = 1 # for div by zero prevention
    epochs = args.epochs;
    last_reported_image_count = 0
    first_pass = True
    ilimit = 0
    train_elapsed = 0
    train_interval = 0
    last_delta = 1
    train_left = 1
    last_train = 0
    
    try:
        device = torch.device("cuda" if use_gpu else "cpu")
        # torch.set_num_threads(os.cpu_count())

        if args.limit > 0:
            ilimit = args.limit
            print("Set limit to " + str(ilimit))
            total_images = epochs * ilimit
            
        logging.info("image_count, train_elapsed, train_interval, content_loss, style_loss, total_loss, reporting_line, train_completion, total_images, train_eta, train_left, delta_time")

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.force_size:
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
        train_loader = DataLoader(train_dataset, batch_size=trial_batch_size)

        if ilimit == 0:
            total_images = epochs * len(train_dataset)
            
        # GPU is unused at this point

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
        style = utils.load_image(args.style_image, args.style_scale)
        style = style_transform(style)
        style = style.repeat(trial_batch_size, 1, 1, 1).to(device)
        
        features_style = vgg(utils.normalize_batch(style))
        gram_style = [utils.gram_matrix(y) for y in features_style]


        if first_pass:
            train_start = time.time()
            first_pass = False

        for e in range(epochs):
            transformer.train()
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0
            ckpt_id = 0
            
            for batch_id, (x, _) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()

                image_count += n_batch
                
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

                train_elapsed = time.time() - train_start
                train_interval = train_elapsed - train_reported
                if train_interval > reporting_interval:
                    reporting_line += 1
                    train_reported = train_elapsed
                    train_completion = image_count / total_images
                    last_reported_image_count = image_count
                    train_eta = 0
                    if last_train == 0:
                        last_train = train_elapsed
                    else:
                        last_train = train_left
                    train_delta = 1
                    
                    if train_completion > 0:
                        train_eta = train_elapsed / train_completion
                        train_left = train_eta - train_elapsed
                        if last_train == 0:
                            train_delta = 0
                        else:
                            train_delta = 1 - (train_left / last_train)

                    mesg = str(image_count) + ", " \
                        + str(train_elapsed) + ", " \
                        + str(train_interval) + ", " \
                        + str(round(agg_content_loss / (batch_id + 1))) + ", " \
                        + str(round(agg_style_loss / (batch_id + 1))) + ", " \
                        + str(round((agg_content_loss + agg_style_loss) / (batch_id + 1))) \
                        + ", " + str(reporting_line) \
                        + ", " + str(train_completion) \
                        + ", " + str(total_images) \
                        + ", " + str(train_eta) \
                        + ", " + str(train_left) \
                        + ", " + str(train_delta)

                    if (args.logfile != ""):
                        logging.info(mesg)
                    print(mesg)

                
                if (args.checkpoint_model_dir != "") and (batch_id + 1) % args.checkpoint_interval == 0:
                    transformer.eval().cpu()
                    ckpt_model_filename = "ckpt_" + str(ckpt_id + 1).zfill(4) + ".pth"
                    ckpt_id += 1;
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save(transformer.state_dict(), ckpt_model_path)
                    transformer.to(device).train()
                    
                if (args.limit != 0 and count >= ilimit) or abort_flag:
                    if abort_flag:
                        print("Aborting run")
                    else:
                        pass # print("Limit reached : " + str(ilimit));
                    break;
                    
                abort_flag = check_abort()

    except:
        print("Stopping run - please wait")
        # print(e)
        abort_flag = False
        except_flag = True
    finally:
        if not except_flag:
            # save model
            transformer.eval().cpu()
            # save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(args.content_weight) + "_" + str(args.style_weight) + ".model"
            save_model_filename = args.model_name + args.model_ext
            save_model_path = os.path.join(args.model_dir, save_model_filename)
            torch.save(transformer.state_dict(), save_model_path)

            if last_reported_image_count != image_count:
                reporting_line += 1
                train_reported = train_elapsed
                train_completion = image_count / total_images
                last_reported_image_count = image_count
                train_eta = 0
                train_left = 0
                if train_completion > 0:
                    train_eta = train_elapsed / train_completion
                    train_left = train_eta - train_elapsed
                
                mesg = str(image_count) + ", " \
                    + str(train_elapsed) + ", " \
                    + str(train_interval) + ", " \
                    + str(round(agg_content_loss / (batch_id + 1))) + ", " \
                    + str(round(agg_style_loss / (batch_id + 1))) + ", " \
                    + str(round((agg_content_loss + agg_style_loss) / (batch_id + 1))) \
                    + ", " + str(reporting_line) \
                    + ", " + str(train_completion) \
                    + ", " + str(total_images) \
                    + ", " + str(train_eta) \
                    + ", " + str(train_left)

                logging.info(mesg)
                print("==> " + mesg)

            print("\nDone, trained model saved at", save_model_path)
            print("Batch size =", trial_batch_size, "- Epochs =", epochs)
            
        #    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

def stylize(args, use_gpu):
    device = torch.device("cuda" if use_gpu else "cpu")
    
    if args.content_image_raw == '':
        content_image = utils.load_image(args.content_image, args.content_scale)
    else:
        content_image = content_image_raw

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
            if args.add_model_ext == 1:
                state_dict = torch.load(os.path.join(args.model_dir, args.model + args.model_ext))
            else:
                state_dict = torch.load(os.path.join(args.model_dir, args.model))
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            if False:
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

