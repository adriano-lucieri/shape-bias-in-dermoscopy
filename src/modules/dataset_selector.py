from Custom_Datasets.Derm7pt import Derm7pt
from Custom_Datasets.Derm7pt_Segmentations import Derm7pt_Segmentations
from Custom_Datasets.Derm7pt_Augmentations import Derm7pt_Augmentations
from Custom_Datasets.ISICb_NV_MEL import ISICb_NV_MEL
from Custom_Datasets.ISICb_NV_MEL_Segmentations import ISICb_NV_MEL_Segmentations
from Custom_Datasets.ISICb_NV_MEL_Augmentations import ISICb_NV_MEL_Augmentations
from Custom_Datasets.ISIC_multiclass import ISIC_multiclass
from Custom_Datasets.ISIC_multiclass_Segmentations import ISIC_multiclass_Segmentations
from Custom_Datasets.ISIC_multiclass_Augmentations import ISIC_multiclass_Augmentations
from Custom_Datasets.Imagenette import Imagenette
from Custom_Datasets.Imagewoof import Imagewoof


def dataset_selector(dataset_name, mode, transforms, seed=42, basepath='/home/lucieri/Datasets/', **kwargs):

    if dataset_name == 'Derm7pt':
        dataset = Derm7pt(split=mode, target_label=kwargs['target_label'], basepath=basepath, transforms=transforms, seed=seed)
    elif dataset_name == 'Derm7pt-seg':
        dataset = Derm7pt_Segmentations(split=mode, target_label=kwargs['target_label'], basepath=basepath, transforms=transforms, seed=seed)
    elif dataset_name == 'Derm7pt-aug':
        dataset = Derm7pt_Augmentations(split=mode, target_label=kwargs['target_label'], augmentation_name=kwargs['augmentation_name'], basepath=basepath, transforms=transforms, seed=seed)
    elif dataset_name[:12] == 'ISICb_NV_MEL':
        if dataset_name == 'ISICb_NV_MEL-seg':
            dataset = ISICb_NV_MEL_Segmentations(mode=mode, transforms=transforms, seed=seed, basepath=basepath)
        elif dataset_name == 'ISICb_NV_MEL-aug':
            dataset = ISICb_NV_MEL_Augmentations(mode=mode, transforms=transforms, seed=seed, basepath=basepath, augmentation_name=kwargs['augmentation_name'])
        else:
            # Get train subsample portion from identifier after dataset name like 'ISICb_NV_MEL_XX'
            portion = int(dataset_name.split('_')[-1]) if len(dataset_name) > 12 else None
            dataset = ISICb_NV_MEL(mode=mode, transforms=transforms, seed=seed, basepath=basepath, portion=portion)
        
    elif dataset_name == 'ISIC_multiclass':
        dataset = ISIC_multiclass(mode=mode, transforms=transforms, seed=seed, basepath=basepath)
    elif dataset_name == 'ISIC_multiclass-seg':
        dataset = ISIC_multiclass_Segmentations(mode=mode, transforms=transforms, seed=seed, basepath=basepath)
    elif dataset_name == 'ISIC_multiclass-aug':
        dataset = ISIC_multiclass_Augmentations(mode=mode, transforms=transforms, seed=seed, basepath=basepath, augmentation_name=kwargs['augmentation_name'])
    elif dataset_name == 'Imagenette':
        dataset = Imagenette(mode=mode, transforms=transforms, seed=seed, basepath=basepath)
    elif dataset_name == 'Imagewoof':
        dataset = Imagewoof(mode=mode, transforms=transforms, seed=seed, basepath=basepath)
    else:
        print(f'No dataset defined for {dataset_name}.')
        raise NotImplementedError

    print('{} length: {}'.format(mode, len(dataset)))

    return dataset


def get_normalization_values(dataset_name):

    if dataset_name == 'PH2':
        mean = [0.7566, 0.5742, 0.5078]
        std = [0.1923, 0.1863, 0.1955]
    elif dataset_name in ['Derm7pt']:
        mean = [0.7579, 0.6583, 0.5911]
        std = [0.2492, 0.2679, 0.2739]
    elif dataset_name in ['ISIC_binary', 'ISIC_multiclass']:
        mean = [0.7029, 0.5767, 0.5695]
        std = [0.1615, 0.1760, 0.1958]
    elif dataset_name == 'ISICb_NV_MEL':
        mean = [0.7024, 0.5214, 0.5224]
        std = [0.1984, 0.1913, 0.2019]
    elif dataset_name == 'ISIC_multiclass':
        mean = [0.6809, 0.5183, 0.5192]
        std = [0.2126, 0.1965, 0.2068]
    else:
        print(f'No normalization values defined for {dataset_name}.')
        raise NotImplementedError

    return mean, std