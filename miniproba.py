import torch
import torch.nn.functional as F

save_dir = '/fhome/gia07/Image_captioning/src/Dataset/roi_features_fastercnn_coco/'
mobilnet_features_name = {'train': 'mobilenet_train.pt', 'valid': 'mobilenet_val.pt', 'test': 'mobilenet_test.pt'}['test']
mobilnet_features = torch.load(save_dir + mobilnet_features_name, map_location=torch.device('cpu'))

print(mobilnet_features[0].shape)
s = 1000 - mobilnet_features[0].shape[0]
a = F.pad(mobilnet_features[0], (0, 0, 0, s))
print(a.shape)