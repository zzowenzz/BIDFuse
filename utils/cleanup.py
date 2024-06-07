import torch.distributed as dist

def cleanup():
    dist.destroy_process_group()