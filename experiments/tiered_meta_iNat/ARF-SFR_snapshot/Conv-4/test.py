import sys
import os
import copy
import torch
import yaml

from trainers import trainer_snapshot_n

sys.path.append('../../../../')
from models.ARFSFR_snapshot import ARFSFR
from utils import util
from trainers.eval_snapshot_n import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'tiered_meta_iNat/test')

args = trainer_snapshot_n.train_parser()
gpu = 0
torch.cuda.set_device(gpu)


stage = 5
epoch = 30


model = ARFSFR(resnet=False)
model.cuda()
models = [copy.deepcopy(model) for i in range(stage)]

[models[i].load_state_dict(torch.load(str(epoch*(i+1))+'_'+'model_Conv-4.pth')) for i in range(stage)]
[m.eval() for m in models]

with torch.no_grad():
    way = 5
    for shot in [1,5]:
        mean,interval = meta_test(data_path=test_path,
                                models=models,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000,
                                epoch=args.epoch,
                                hw_range=args.hw_range
                                  )
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))