import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='pretrained model path')
    parser.add_argument("--test_weight_path", default=None, type=str, help='testing on benchmark')
    parser.add_argument('--batch_size', default=1, type=int) ## 10
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--decoder_type', default='PG-DUN', type=str, help='reconstruction network type')
    parser.add_argument('--cr', default=0.1, type=float)  # 0.01, 0.04, 0.10, 0.25, 0.5
    parser.add_argument('--size', default=[800, 800])  ## 800
    parser.add_argument('--meas_size', default=[800, 800])  # 200
    parser.add_argument('--matrix_train', default=True, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--color_channels', default=10, type=int)  ## 10
    parser.add_argument('--stages', default=6, type=int)  ## 5
    parser.add_argument('--channels', default=64, type=int)
    parser.add_argument('--mid_blocks', default=1)
    parser.add_argument('--enc_blocks', default=[1, 1])
    parser.add_argument('--dec_blocks', default=[1, 1])
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=1, type=int)
    parser.add_argument('--test_flag', default=True, type=bool)
    parser.add_argument('--train_data_path', type=str, default='/path/to/train-dataset')  
    parser.add_argument('--benchmark_path', type=str, default='/path/to/test—dataset') 
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor', help="Enable compilation w/ specified backend (default: inductor).")
    args = parser.parse_args()
    return args
