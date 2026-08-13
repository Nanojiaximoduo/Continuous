import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision_local.inter_models.architecture import *
import PIL
import torchvision.transforms as transforms
import cv2

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # 加载模型
    # pretrained_model_path = opt.pretrained_model_path
    pretrained_model_path = r'/home/xiaozhicheng/continues/model_zoo/mst_plus_plus.pth'
    # method = opt.method
    method = 'mst_plus_plus'
    channels = 31
    model = model_generator(method, pretrained_model_path, channels)

    model = torch.nn.DataParallel(model).cuda()
    image = PIL.Image.open(r'cat_test.jpg').convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False)
    ])
    image = transform(image)
    result = model(torch.ones(4, 3, 224, 224).cuda())

    arr = result.cpu().detach().numpy()
    # for i in range(31):
    #     print(i)
    #     channel = arr[0][i]  # 提取第 i 个通道
    #     filename = os.path.join(r'C:\Users\Administrator\PycharmProjects\intellegenthub\fasttrain\out', f"channel_{i:02d}.png")
    #     cv2.imwrite(filename, channel*255)
    # print(result.size())
