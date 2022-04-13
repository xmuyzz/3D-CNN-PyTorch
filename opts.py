import argparse


def parse_opts():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')
    parser.add_argument('--cnn_name', default='ResNet', type=str, help='cnn model names')
    parser.add_argument('--model_depth', default=101, type=str, help='model depth (18|34|50|101|152|200)')
    parser.add_argument('--n_classes', default=2, type=str, help='model output classes')
    parser.add_argument('--in_channels', default=1, type=str, help='model input channels (1|3)')
    parser.add_argument('--sample_size', default=128, type=str, help='image size')

    args = parser.parse_args()

    return args
