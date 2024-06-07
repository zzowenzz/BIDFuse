import torch.distributed as dist
import logging

def log_from_gpu(msg):
    """
    Use rank 0 to log the message;
    """
    if dist.is_initialized() and dist.get_rank() == 0:
            logging.info(msg)