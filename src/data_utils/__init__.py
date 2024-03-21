import importlib
from os import path as osp
try:
    from data_utils.utils import scandir
except ImportError:
    from src.data_utils.utils import scandir

__all__ = ['create_dataset']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('dataset.py')
]


try: 
    dataset_modules = [importlib.import_module(f'data_utils.{file_name}')
                    for file_name in dataset_filenames]
except ImportError:
    dataset_modules = [importlib.import_module(f'src.data_utils.{file_name}')
                    for file_name in dataset_filenames]

def create_dataset(data, partitions, TOTAL_MAX_WORDS, word2idx, train, augment, dataset_name, dataset_type='Data_word', device = None, size=224, type_partition = None):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """ 

    # dynamic instantiation  如何理解动态实例   化？逐个遍历已经import进来的包，就是所有dataset.py 
    for module in dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)  #getattr() 获取对象的属性值
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(data = data, partition = partitions, train = train, augment = augment, TOTAL_MAX_WORDS = TOTAL_MAX_WORDS, word2idx = word2idx, device = device, size=size,  type_partition = type_partition)

    print(f'Dataset {dataset.__class__.__name__} - {dataset_name} '
        'is created.')
    return dataset