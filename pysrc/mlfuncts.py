from delphifuncts import *
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
import json
import utils
from transformer_net import TransformerNet
from vgg import *
import logging

have_psutils = True
try:
    import psutil
except Exception as e:
    print(e)
    print('Exception type is:', e.__class__.__name__)
    have_psutils = False

def get_gpu_memory(have_psutils, use_gpu):
    if use_gpu:
        d = torch.cuda.get_device_name(0)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved

        gpu = TJsonLog(
            device = d,
            free = f,
            reserved = r,
            allocated = a,
            total = t)

    if have_psutils:
        m = psutil.virtual_memory()
        mem = TJsonLog(
            total = m.total,
            available = m.available,
            percent = m.percent,
            used = m.used,
            free = m.free)

    if have_psutils and use_gpu:
        stats = TJsonLog(gpu = gpu, mem = mem)
    elif have_psutils and not use_gpu:
        stats = TJsonLog(gpu = False, mem = mem)
    elif not have_psutils and use_gpu:
        stats = TJsonLog(gpu = gpu, mem = False)
    else:
        stats = TJsonLog(gpu = False, mem = False)

    return(stats)

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
    if have_delphi_io:
        ioopts = TDelphiInputOutput();

    abort_flag = False
    except_flag = False
    train_sample_flag = False
    e = 0
    agg_content_loss = 0
    agg_style_loss = 0.
    count = 0
    image_count = 0
    batch_id = 0
    reporting_interval = 1
    train_start = time.time()
    train_reported = 0
    reporting_line = 0
    total_images = 1 # for div by zero prevention
    epochs = args.epochs;
    last_reported_image_count = 0
    ilimit = 0
    train_elapsed = 0
    train_interval = 0
    last_delta = 1
    train_left = 1
    last_train = 0

    try:
        device = torch.device("cuda" if use_gpu else "cpu")
        # torch.set_num_threads(os.cpu_count())

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

        if args.limit > 0:
            ilimit = args.limit
            total_images = ilimit
            epochs = (total_images // len(train_dataset)) + 1
        else:
            total_images = epochs * len(train_dataset)

        # GPU is unused at this point

        transformer = TransformerNet(showTime = False).to(device)

        optimizer = Adam(transformer.parameters(), args.lr)
        mse_loss = torch.nn.MSELoss()

        if args.net.casefold() == 'vgg16':
            if have_delphi_train:
                vgg = Vgg16(requires_grad=False, vgg_path = args.vgg16_path).to(device)
            else:
                vgg = Vgg16(requires_grad=False).to(device)
        else:
            if have_delphi_train:
                vgg = Vgg19(requires_grad=False, vgg_path = args.vgg19_path).to(device)
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


        train_start = time.time()

        print("Set ilimit = " + str(ilimit) + ", epochs = " + str(epochs) + ", total_images = " + str(total_images))

        for e in range(epochs):
            if ((ilimit != 0) and (image_count >= ilimit)) or abort_flag:
                if abort_flag:
                    print("Epoch aborting run !!!")
                else:
                    print("Epoch limit reached : " + str(ilimit));
                break;

            transformer.train()
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0

            for batch_id, (x, _) in enumerate(train_loader):
                if ((ilimit != 0) and (image_count >= ilimit)) or abort_flag:
                    if abort_flag:
                        print("Batch aborting run !!!")
                    else:
                        print("Batch limit reached : " + str(ilimit));
                    break

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
                            train_delta = 1
                        else:
                            train_delta = 1 - (train_left / last_train)

                    if(args.log_event_api):
                        # system = get_gpu_memory(have_psutils, use_gpu)

                        if have_delphi_train:
                            ptrain.TrainProgress(TJsonLog(
                                image_count = image_count,
                                train_elapsed = round(train_elapsed),
                                train_interval = train_interval,
                                content_loss = round(agg_content_loss / (batch_id + 1)),                    # Loss chart
                                style_loss = round(agg_style_loss / (batch_id + 1)),                        # Loss chart
                                total_loss = round((agg_content_loss + agg_style_loss) / (batch_id + 1)),   # Loss chart
                                reporting_line = reporting_line,        # Enable Buttons when = 1
                                train_completion = train_completion,    # Progress bar
                                total_images = total_images,
                                train_eta = round(train_eta),           # ETA Progress
                                train_left = round(train_left),         # ETA Progress
                                train_delta = train_delta               # Enable Countdown
                                ))
                        if have_delphi_io:
                            abort_flag = ioopts.TrainAbortFlag;
                            train_sample_flag = ioopts.TrainSampleFlag;
                            ioopts.TrainSampleFlag = False;
                            
                        if not have_delphi_train:
                            print(json.dumps(TJsonLog(
                                image_count = image_count,
                                train_elapsed = round(train_elapsed),
                                train_interval = train_interval,
                                content_loss = round(agg_content_loss / (batch_id + 1)),
                                style_loss = round(agg_style_loss / (batch_id + 1)),
                                total_loss = round((agg_content_loss + agg_style_loss) / (batch_id + 1)),
                                reporting_line = reporting_line,
                                train_completion = train_completion,
                                total_images = total_images,
                                train_eta = round(train_eta),
                                train_left = round(train_left),
                                train_delta = train_delta
                                )))

                    if (args.logfile != ""):
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
                        logging.info(mesg)

                if train_sample_flag:
                    train_sample_flag = False
                    transformer.eval().cpu()
                    ckpt_model_filename = args.model_name + "-ckpt-" + str(e)
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save(transformer.state_dict(), ckpt_model_path + args.model_ext)
                    print("Saved", ckpt_model_path + args.model_ext)
                    sample = stylize(TStylize( content_image = "input-images/haywain.jpg",
                        content_image_raw = "",
                        output_image = args.checkpoint_model_dir + "/" + ckpt_model_filename + ".jpg",
                        model = ckpt_model_filename,
                        model_dir = args.checkpoint_model_dir,
                        model_ext = ".pth",
                        logfile = "",
                        content_scale = 1,
                        ignore_gpu = False,
                        export_onnx = False,
                        add_model_ext = True,
                        log_event_api = False), False)
                    print("Sample =", sample);
                    # ioopts.SampleFilename = sample;
                    # pinout.SampleToDelphi()
                    transformer.to(device).train()

    except Exception as e:
        print(e)
        print('Exception type is:', e.__class__.__name__)
        # print("Stopping run - please wait")
        abort_flag = False
        except_flag = True
        raise e
    finally:
        if not except_flag:
            # save model
            transformer.eval().cpu()
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
                train_delta = 0
                if train_completion > 0:
                    train_eta = train_elapsed / train_completion
                    train_left = train_eta - train_elapsed

                if (args.logfile != ""):
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

                print(json.dumps(TJsonLog(
                    image_count = image_count,
                    train_elapsed = round(train_elapsed),
                    train_interval = train_interval,
                    content_loss = round(agg_content_loss / (batch_id + 1)),
                    style_loss = round(agg_style_loss / (batch_id + 1)),
                    total_loss = round((agg_content_loss + agg_style_loss) / (batch_id + 1)),
                    reporting_line = reporting_line,
                    train_completion = train_completion,
                    total_images = total_images,
                    train_eta = round(train_eta),
                    train_left = round(train_left),
                    train_delta = train_delta
                    )))

            print("\nDone, trained model saved at", save_model_path)
            print("Batch size =", trial_batch_size, "- Epochs =", epochs)
            if have_delphi_train:
                return(json.dumps(TJsonLog(
                    image_count = image_count,
                    train_elapsed = round(train_elapsed),
                    train_interval = train_interval,
                    content_loss = round(agg_content_loss / (batch_id + 1)),
                    style_loss = round(agg_style_loss / (batch_id + 1)),
                    total_loss = round((agg_content_loss + agg_style_loss) / (batch_id + 1)),
                    reporting_line = reporting_line,
                    train_completion = train_completion,
                    total_images = total_images,
                    train_eta = round(train_eta),
                    train_left = round(train_left),
                    train_delta = train_delta
                    )))

        #    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

def stylize(args, use_gpu):
    device = torch.device("cuda" if use_gpu else "cpu")

    if args.content_image_raw == "":
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
            style_time = time.time();
            if have_delphi_style:
                pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'StartStyle', time = 0))
            style_model = TransformerNet(showTime = True)
            if args.add_model_ext == 1:
                state_dict = torch.load(os.path.join(args.model_dir, args.model + args.model_ext))
            else:
                state_dict = torch.load(os.path.join(args.model_dir, args.model))
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()
                if have_delphi_style:
                    pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'EndStyle', time = time.time() - style_time))
            else:
                output = style_model(content_image).cpu()
                if have_delphi_style:
                    pstyle.StyleProgress(TJsonLog(event = 'styleTime', subevent = 'EndStyle', time = time.time() - style_time))
    utils.save_image(args.output_image, output[0])
    return (args.output_image)

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

class TStylize(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

##### Test #####
#opts = TStylize( content_image = "..\\input-images\\haywain.jpg",
#    content_image_raw = "",
#    output_image = "..\\output-images\\command-test.jpg",
#    model = "dae_mosaic_1-200",
#    model_dir = "..\\models",
#    model_ext = ".pth",
#    logfile = "",
#    content_scale = 1,
#    ignore_gpu = False,
#    export_onnx = False,
#    add_model_ext = True,
#    log_event_api = False)
#stylize(opts, False)
##### Test #####
