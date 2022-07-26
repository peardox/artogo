import os
import sys
try:
    from delphifuncts import *
    print("Delphi OK")
except Exception as e:
    # Most likely running from command line
    have_delphi = False
    sys.path.append('pysrc')
    from delphifuncts import *
    print("Delphi Missing")
   
import time
import logging
import torch
import json
from mlfuncts import *

have_psutils = True
try:
    import psutil
except Exception as e:
    have_psutils = False

def get_sys_info():
    if gpu_supported:
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

    if have_psutils and gpu_supported:
        stats = TJsonLog(gpu = gpu, mem = mem)
    elif have_psutils and not gpu_supported:
        stats = TJsonLog(gpu = False, mem = mem)
    elif not have_psutils and gpu_supported:
        stats = TJsonLog(gpu = gpu, mem = False)
    else:
        stats = TJsonLog(gpu = False, mem = False)
        
    return(stats)
    

def check_gpu():
    gpu_supported = False
    try:
        torch.cuda.init()
        if(torch.cuda.is_available()):
            gpu_supported = True
    except:
        pass

    return gpu_supported

def show_elapsed(from_time):
    elapsed = time.time() - from_time
    print("Elapsed time = %f secs" % (elapsed))
    hour = elapsed // 3600
    elapsed %= 3600
    minutes = elapsed // 60
    elapsed %= 60
    seconds = elapsed
    print("Elapsed time = %d hours %d mins %d secs" % (hour, minutes, seconds))

def do_train(opts = None):
    is_gpu_available = check_gpu()
    
    if opts == None:
        opts = TTrain(dataset="/train/unsplash/256",
            style_image="style-images/gig.jpg",
            model_name="gig-256",
            model_dir="models",
            checkpoint_model_dir="cache",
            model_ext = ".pth",
            net="vgg19",
            vgg16_path=None,
            vgg19_path=None,
            logfile="",
            epochs=2,
            limit=0,
            batch_size=8,
            image_size=256,
            seed=42,
            content_weight=1e5,
            style_weight=1e10,
            lr=1e-3,
            style_scale=1.0,
            channels=32,
            force_size=True,
            ignore_gpu=False,
            log_event_api=False)

#    check_paths(args)
    trial_batch = opts.batch_size
    start = time.time()

    while(1):
        oom = False
        try:
            print("Trying batch of ", trial_batch)
            if opts.ignore_gpu:
                train(opts, False, trial_batch)
            else:
                train(opts, is_gpu_available, trial_batch)
        except RuntimeError as e:
            print("Hit exception handler")
            if trial_batch > 0:
                oom = True
            else:
                print(e)
                return(1)
        else:
            break

        if oom:
            trial_batch -= 1
            if is_gpu_available and not opts.ignore_gpu:
                torch.cuda.empty_cache()
            if trial_batch == 0:
                print("No batch size found to run current training session (style image too large)")
                return(1)

    show_elapsed(start)
    
def do_stylize(opts = None):
    is_gpu_available = check_gpu()
    
    if opts == None:
        opts = TStylize( content_image = "input-images/haywain.jpg",
            content_image_raw = "",
            output_image = "output-images/command-test.jpg",
            model = "dae_mosaic_1-200",
            model_dir = "models",
            model_ext = ".pth",
            logfile = "",
            content_scale = 1,
            ignore_gpu = True,
            export_onnx = False,
            add_model_ext = True,
            log_event_api = False)

    start = time.time()
    if opts.ignore_gpu:
        stylize(opts, False)
    else:
        stylize(opts, is_gpu_available)
    show_elapsed(start)

def do_test(opts = None):
    # is_gpu_available = check_gpu()
    
    if opts == None:
        opts = TStylize( content_image = "input-images\\haywain.jpg",
            content_image_raw = "",
            output_image = "output-images\\test-dae-sketch1-512.jpg",
            model = "test-dae-sketch1-512",
            # model = "mosaic-vgg16-1010-512",
            model_dir = "models",
            model_ext = ".pth",
            logfile = "",
            content_scale = 1,
            ignore_gpu = False,
            export_onnx = False,
            add_model_ext = True,
            log_event_api = False
            )

    for k, v in opts.items():
        print(k, '=', v)

def delphi_train():
    is_gpu_available = check_gpu()

    trainopts = TDelphiTrain()

    for i in ptrain.GetPropertyList():
        print(i, '=', ptrain.GetProperty(i))
    
    rval = None
    
    trial_batch = trainopts.batch_size
    
    start = time.time()
    
    while(1):
        oom = False
        try:
            print("Trying batch of ", trial_batch)
            if trainopts.ignore_gpu:
                rval = train(trainopts, False, trial_batch)
            else:
                rval = train(trainopts, is_gpu_available, trial_batch)
        except RuntimeError as e:
            print("Hit exception handler")
            if trial_batch > 0:
                oom = True
            else:
                print(e)
                return("Unrecoverable Error")
        else:
            break

        if oom:
            trial_batch -= 1
            if is_gpu_available and not trainopts.ignore_gpu:
                torch.cuda.empty_cache()
            if trial_batch == 0:
                return("No batch size found to run current training session (style image too large)")

    show_elapsed(start)
    return (rval)
    
def delphi_style():
    is_gpu_available = check_gpu()

    styleopts = TDelphiStylize()

    for i in pstyle.GetPropertyList():
        print(i, '=', pstyle.GetProperty(i))

    start = time.time()

    if styleopts.ignore_gpu:
        rval = stylize(styleopts, False)
    else:
        rval = stylize(styleopts, is_gpu_available)

    show_elapsed(start)
    return (rval)
    
class TStylize(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
class TTrain(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class TProperties:
    def __getattr__(Self, Key):
        return props.GetProperty(Key)

    def __setattr__(Self, Key, Value):
        props.SetProperty(Key, Value)

    def __repr__(Self):
        tmp = ""
        for i in props.GetPropertyList():
            if tmp:
                tmp = tmp + ", "
            tmp = tmp + i + " = " + str(getattr(Self,i))
        return tmp

def do_main():
   print("Running from command line");
   do_stylize()
    
try:
    if not __embedded_python__:
        do_main()
except NameError:
    # Only run main if called explicitly
    if __name__ == "__main__":
        do_main()
else:
    pass
#    gpu_supported = check_gpu()
#    print("Using Embedded Environment")
#    print(json.dumps(get_sys_info()))

#    styleopts = TDelphiStylize()
#    for i in pstyle.GetPropertyList():
#        print(i, '=', pstyle.GetProperty(i))


#    trainopts = TDelphiTrain()
#    for i in ptrain.GetPropertyList():
#        print(i, '=', ptrain.GetProperty(i))
