
import sys, resource, torch
from torchvision import models
import torch.nn as nn
nama, ckpt, n, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
if mode == 'baseline':
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss); sys.exit()
if nama=='MobileNetV3-Small':
    m=models.mobilenet_v3_small(); m.classifier[3]=nn.Linear(m.classifier[3].in_features,n)
elif nama=='EfficientNet-B0':
    m=models.efficientnet_b0(); m.classifier[1]=nn.Linear(m.classifier[1].in_features,n)
else:
    m=models.shufflenet_v2_x1_0(); m.fc=nn.Linear(m.fc.in_features,n)
m.load_state_dict(torch.load(ckpt, map_location='cpu')); m.eval()
with torch.no_grad():
    for _ in range(20): m(torch.randn(1,3,224,224))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
