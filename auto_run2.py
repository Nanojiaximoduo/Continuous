import subprocess
import subprocess
# --dataset cifar10 --cuda_id 0 -a mobilenet_v3_small --threshold_factor 0.2 --fast True
params_list = [
    [
        '/data/Datasets/imagenet/',
        '31',
        '2',
        '--method',
        'mst',
        '--cuda_id',
        '0,1,2,3,4,5,6,7',
        '--track_url',
        'http://10.6.254.127:5000',
        '--dataset',
        'imagenet',
        '--pretrained_model_path',
        '/home/xiaozhicheng/continues/model_zoo/mst.pth',
        '-b',
        '256',
        '--multiprocessing-distributed',
        '--rank',
        '0',
        '--dist-url',
        'tcp://127.0.0.1:23459',
        '--world-size',
        '1',
        '-j',
        '1',
        '--times',
        f'{i}'
    ] for i in range(1)
]
print(params_list)
for i, params in enumerate(params_list):
    subprocess.run(["python", "main2.py"] + params)
