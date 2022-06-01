import torch
import pynvml

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
