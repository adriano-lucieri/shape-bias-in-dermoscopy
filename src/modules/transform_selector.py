import cv2
import torch

import numpy as np

from dataset_selector import get_normalization_values


class replace_phase(object):
    """Custom transform replacing the phase spectrum of a given RGB image
    with the respective spectrum of a noise or IID image.

    Args:
        object (_type_): _description_
    """
    def __init__(self, mode='fixed', image_size=224):

        self.mode = mode
        self.image_size = image_size

        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(2147483647)

        self.replace_image = None    

    def get_replace_image(self, img):

        if self.mode == 'fixed':
            # Generates only once
            if self.replace_image is None:
                self.replace_image = 2 * torch.rand(3, self.image_size, self.image_size, generator=self.generator)-1 
            else:
                pass
        elif self.mode == 'random':
            # Generates new image on each call
            self.replace_image = 2 * torch.rand(3, self.image_size, self.image_size, generator=self.generator)-1
        elif self.mode == 'iid':
            # Sets first iid image on first call.
            if self.replace_image is None:
                self.replace_image = img.clone()
            else:
                pass
        else:
            raise NotImplementedError


    def get_amplitude_phase(self,image):

        img_fft = torch.fft.rfft2(image.clone())
        img_amp = torch.abs(img_fft)
        img_phase = torch.angle(img_fft)

        return img_amp, img_phase

    def reconstruct(self, amp, phase):

        reconstruction = amp * torch.exp(1j * phase)
        reconstruction = torch.fft.irfft2(reconstruction)

        return reconstruction

    def replace_phase(self, img, rand_img):

        org_amp, _ = self.get_amplitude_phase(img)
        _, rand_phase = self.get_amplitude_phase(rand_img)
        new_img = self.reconstruct(org_amp, rand_phase)

        return new_img


    def __call__(self, img):

        self.get_replace_image(img)
        img = self.replace_phase(img, self.replace_image)
        return img
      
class replace_amplitude(object):
    """Custom transform replacing the amplitude spectrum of a given RGB image
    with the respective spectrum of a noise or IID image.

    Args:
        object (_type_): _description_
    """
    def __init__(self, mode='fixed', image_size=224):

        self.mode = mode
        self.image_size = image_size

        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(2147483647)

        self.replace_image = None    

    def get_replace_image(self, img):

        if self.mode == 'fixed':
            # Generates only once
            if self.replace_image is None:
                self.replace_image = 2 * torch.rand(3, self.image_size, self.image_size, generator=self.generator)-1 
            else:
                pass
        elif self.mode == 'random':
            # Generates new image on each call
            self.replace_image = 2 * torch.rand(3, self.image_size, self.image_size, generator=self.generator)-1
        elif self.mode == 'iid':
            # Sets first iid image on first call.
            if self.replace_image is None:
                self.replace_image = img.clone()
            else:
                pass
        else:
            raise NotImplementedError

    def get_amplitude_phase(self,image):

        img_fft = torch.fft.rfft2(image.clone())
        img_amp = torch.abs(img_fft)
        img_phase = torch.angle(img_fft)

        return img_amp, img_phase
    
    def replace_amplitude(self, img, rand_img):

        _, org_phase = self.get_amplitude_phase(img)
        rand_ampl, _ = self.get_amplitude_phase(rand_img)
        new_img = self.reconstruct(rand_ampl, org_phase)

        return new_img
        
    def reconstruct(self, amp, phase):

        recon = amp * torch.exp(1j * phase)
        recon = torch.fft.irfft2(recon)

        return recon
        
    def __call__(self, img):
    
        self.get_replace_image(img)
        img = self.replace_amplitude(img, self.replace_image)
        return img

class apr_p(object):
    """APR-P is applied directly on the batch and not as a
    separate transform function.

    Args:
        object (_type_): _description_
    """
    def __call__(self, img):
        return img

class color_removed(object):
    def __init__(self, dataset_mean):
        self.dataset_mean = torch.tensor(dataset_mean)
        self.dataset_mean = self.dataset_mean[:, None, None]

    def create_line_drawing_image(self, img):
        img = img.cpu().numpy()
        img = np.moveaxis(img, 0, -1)
        kernel = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            ], np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_dilated = cv2.dilate(img_gray, kernel, iterations=1)
        img_diff = cv2.absdiff(img_dilated, img_gray)
        contour = 255 - img_diff

        contour = torch.from_numpy(contour)
        contour = torch.stack([contour, contour, contour])
        
        # Multiply with image mean
        contour = contour - contour.min()
        contour = contour / contour.max()
        contour = contour * self.dataset_mean

        return contour

    def __call__(self, img):
        return self.create_line_drawing_image(img)

class scramble_pixels(object):

    def __call__(self, orig_img):

        img = orig_img.reshape((3, -1))

        # Shuffle
        p = np.random.permutation(np.arange(img.shape[-1]))
        img = img[:, p]

        # Resize
        img = img.reshape((orig_img.shape[0], orig_img.shape[1], orig_img.shape[2]))

        return img

class dummy_transform(object):
    """Dummy transform doing nothing.

    Args:
        object (_type_): _description_
    """
    def __call__(self, img):
        return img

class normalize(object):
    """APR-P is applied directly on the batch and not as a
    separate transform function.

    Args:
        object (_type_): _description_
    """
    def __call__(self, img):
        img = img - torch.min(img.reshape(-1))
        img = torch.true_divide(img, (torch.max(img.reshape(-1)) + 1e-7))
        return img

def select_transform(config):

    setup_name = config['dataset_parameters']['preprocessing']
    dataset_name = config['dataset_parameters']['train_dataset']

    print('\n')
    print(40*'#')
    print(f'# Preprocessing Setup: {setup_name:15} #')
    print(40*'#', '\n')

    if setup_name == 'replace_phase':
        kwargs = {
            'mode': config['dataset_parameters']['replacement_mode'],
            'image_size': int(config['dataset_parameters']['image_size'])
        }
        return replace_phase, kwargs
    elif setup_name == 'replace_amplitude':
        kwargs = {
            'mode': config['dataset_parameters']['replacement_mode'],
            'image_size': int(config['dataset_parameters']['image_size'])
        }
        return replace_amplitude, kwargs
    elif setup_name in ['apr_p_rfft', 'af_apr_p', 'mixed_apr_p']:
        kwargs = {}
        return apr_p, kwargs
    elif setup_name == 'none':
        kwargs = {}
        return dummy_transform, kwargs
    elif setup_name == 'color-removed':
        mean, sdt = get_normalization_values(dataset_name=dataset_name)
        kwargs = {'dataset_mean': mean}
        return color_removed, kwargs
    elif setup_name == 'color-only':
        kwargs = {}
        return scramble_pixels, kwargs
    else:
        raise NotImplementedError