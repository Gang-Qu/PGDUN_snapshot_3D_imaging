from thop import profile

from model.Network import PGDUN
from opts import parse_args
import torch

import time
import os
from torch.cuda.amp import autocast, GradScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args = parse_args()
model = PGDUN(imag_size=args.size,
                     meas_size=args.meas_size,
                     img_channels=args.color_channels,
                     channels=args.channels,
                     mid_blocks=args.mid_blocks,
                     enc_blocks=args.enc_blocks,
                     dec_blocks=args.dec_blocks,
                     stages=args.stages,
                     matrix_train=args.matrix_train).to(args.device)

input = torch.randn(1, 1, 800, 800, 10).cuda()
flop, para = profile(model, input)
print('Flops:',"%.2fM" % (flop/1e6), 'Params:',"%.2fM" % (para/1e6))
total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: %.2fM' % (total/1e6))

total_time = 0
num =10
input = torch.randn(1, 800, 800, 10).cuda()
with torch.no_grad():
    for i in range(num):
        torch.cuda.synchronize()
        start = time.time()
        # print('input:', input.shape)
        out = model(input)
        torch.cuda.synchronize()
        end = time.time()
        runtime = end-start
        total_time +=runtime
        # print('forward time: ',runtime)
print('runtime:{:.2f},  para: {:.2f} M, FLOATs: {:.2f} G. '.format(total_time/num,para/1e6,flop/1e9))
print('Model parameter number: {}'.format(len(model_dict)))
