from pathlib  import Path


data_dir=Path('~/datasets/fundus/binary_expanded/selected').expanduser()
#data_dir = Path("clean_data_left/test_2class")
#data_dir=Path('clean_data_left').expanduser()
#data_dir=Path('clean_data_left/test').expanduser()
data_dir=Path('clean_data_left/test_1280_selected_cropped').expanduser()
# print(data_dir)
class_names = [
                'Gradable',
                'Usable',
                'Ungradable',
                ]

base_output_folder = Path("results").expanduser()
base_output_folder.mkdir(exist_ok=True,parents=True)

image_size = 1280#640
normalization_mu =  [0.485, 0.456, 0.406]
normalization_sigma= [0.229, 0.224, 0.225]

 
 
from torchvision import transforms

test_transformation = transforms.Compose([
          transforms.Resize(image_size),
          transforms.CenterCrop(image_size),
          transforms.ToTensor(),
          transforms.Normalize(normalization_mu,normalization_sigma)
      ])

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mu,sigma=(torch.tensor(p).to(device) for p in [normalization_mu,normalization_sigma])

# inverse_normalize_transform = transforms.Normalize(
#     mean=-mu/sigma,
#     std=1/sigma)

inverse_normalize_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = 1/sigma),
                                transforms.Normalize(mean = -mu,
                                                     std = [ 1., 1., 1. ]),
                               ])

normalize_transform = transforms.Normalize(mu,sigma)

inverse_normalize_transform_cpu = transforms.Normalize(
    mean=-mu.cpu()/sigma.cpu(),
    std=1/sigma.cpu())