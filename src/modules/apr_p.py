import torch
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mix_data_rfft(x):
    """APR-P implementation using RFFT2 instead.

    Args:
        x (tensor): batch of input tensors

    Returns:
        tensor: Amplitude-phase recombined batch.
    """
    
    p = random.uniform(0, 1)

    if p > 0.5:
        return x

    batch_size = x.size()[0]
    if DEVICE == 'cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.rfft2(x)
    abs, angle = torch.abs(fft_1), torch.angle(fft_1)

    fft_1 = abs[index, :]*torch.exp((1j) * angle)

    mixed_x = torch.fft.irfft2(fft_1)

    return mixed_x

def af_apr_p(x):
    """Amplitude-focused APR-P implementation using RFFT2.

    Args:
        x (tensor): batch of input tensors

    Returns:
        tensor: Amplitude-phase recombined batch.
    """
    
    p = random.uniform(0, 1)

    if p > 0.5:
        return x

    batch_size = x.size()[0]
    if DEVICE == 'cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.rfft2(x)
    abs, angle = torch.abs(fft_1), torch.angle(fft_1)

    fft_1 = abs*torch.exp((1j) * angle[index, :])

    mixed_x = torch.fft.irfft2(fft_1)

    return mixed_x

def mixed_apr_p(x):
    """Mixed APR-P implementation using RFFT2.

    Args:
        x (tensor): batch of input tensors

    Returns:
        tensor: Amplitude-phase recombined batch.
    """
    
    p = random.uniform(0, 1)
    if p > 0.5:
        return x

    batch_size = x.size()[0]
    if DEVICE == 'cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.rfft2(x)
    abs, angle = torch.abs(fft_1), torch.angle(fft_1)

    p = random.uniform(0, 1)
    if p > 0.5:
        # Amplitude-label
        fft = abs*torch.exp((1j) * angle[index, :])
    else:
        # Phase-label
        fft = abs[index, :]*torch.exp((1j) * angle)

    mixed_x = torch.fft.irfft2(fft)

    return mixed_x

def dummy(x):
    return x

def select_apr_p(setup_name):

    if setup_name == 'apr_p_rfft':
        return mix_data_rfft, dummy
    elif setup_name == 'af_apr_p':
        return af_apr_p, dummy
    elif setup_name == 'mixed_apr_p':
        return mixed_apr_p, dummy
    else:
        return dummy, dummy