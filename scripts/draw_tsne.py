import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision_local
from torchvision_local.models.resnet import b_spline_feature
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.interpolate import BSpline
from scipy.spatial import Voronoi
from matplotlib import colors
import sys
import types
import os
from matplotlib.colors import ListedColormap

# ====================
# 配置参数
# ====================
config = {
    'rgb_model_path': 'model/resnet34_0_imagenet.pth',
    'continuous_model_path': 'model/resnet34_12_imagenet.pth',
    'dataset_name': 'imagenet', 
    'data_dir': '/home/Datasets/imagenet',
    'batch_size': 256,
    'num_samples': 50000,
    'rgb_num_classes': 1000,    
    'cont_num_classes': 1000,
    'feature_dim': 1024,
    'perplexity': 30,
    'n_iter': 1000,
    'dpi': 300,
    'palette_strategy': 'hsv',
    'save_path': './feature_comparison_imagenet.png'
}
def generate_high_dim_colormap(n_classes: int, strategy: str = 'hierarchical_hsv') -> ListedColormap:
    """
    为1000个类别生成高可区分度的色彩映射 [核心改进]
    
    参数:
        n_classes (int): 类别总数
        strategy (str): 色彩生成策略 
            - 'default': Matplotlib默认映射 (n>10时效果差)
            - 'hierarchical_hsv': 分层HSV色彩空间采样 (推荐)
            - 'golden_angle': 黄金分割角算法 (最均匀分布)
            
    返回:
        ListedColormap: 包含n_classes种颜色的自定义色彩映射
    """
    # 黄金角公式 (137.5度)
    GOLDEN_ANGLE = 137.507764050037854 * (np.pi / 180)  # 转换为弧度
    
    if strategy == 'hierarchical_hsv':
        # 分层HSV策略 - 基于色彩层级树
        levels = int(np.ceil(np.log2(n_classes)))
        colors_list = []
        
        # 递归生成HSV色彩
        def gen_colors(level, h_range, s_range, v_range):
            if level == 0:
                if len(colors_list) >= n_classes:
                    return
                    
                # 在给定范围内随机采样
                h = np.random.uniform(h_range[0], h_range[1])
                s = np.random.uniform(s_range[0], s_range[1])
                v = np.random.uniform(v_range[0], v_range[1])
                colors_list.append(colors.hsv_to_rgb([h, s, v]))
            else:
                # 划分HSV空间
                gen_colors(level-1, (h_range[0], h_range[0]+(h_range[1]-h_range[0])/2), 
                          (s_range[0], s_range[0]+(s_range[1]-s_range[0])/2), 
                          (v_range[0], v_range[0]+(v_range[1]-v_range[0])/2))
                
                gen_colors(level-1, (h_range[0], h_range[0]+(h_range[1]-h_range[0])/2), 
                          (s_range[0], s_range[0]+(s_range[1]-s_range[0])/2), 
                          (v_range[0]+(v_range[1]-v_range[0])/2, v_range[1]))
                          
                gen_colors(level-1, (h_range[0]+(h_range[1]-h_range[0])/2, h_range[1]), 
                          (s_range[0]+(s_range[1]-s_range[0])/2, s_range[1]), 
                          (v_range[0], v_range[0]+(v_range[1]-v_range[0])/2))
                          
                gen_colors(level-1, (h_range[0]+(h_range[1]-h_range[0])/2, h_range[1]), 
                          (s_range[0], s_range[0]+(s_range[1]-s_range[0])/2), 
                          (v_range[0]+(v_range[1]-v_range[0])/2, v_range[1]))
        
        # 初始调用
        gen_colors(levels, (0, 0.95), (0.5, 1.0), (0.5, 1.0))
        
    elif strategy == 'golden_angle':
        # 黄金分割角策略 - 确保最均匀分布
        hues = [(i * GOLDEN_ANGLE) % (2 * np.pi) / (2 * np.pi) for i in range(n_classes)]
        colors_list = [colors.hsv_to_rgb([h, 0.8, 0.9]) for h in hues]
        
    else:
        # 默认策略 - 线性分布HSV
        hues = np.linspace(0, 0.9, n_classes, endpoint=False)
        colors_list = [colors.hsv_to_rgb([h, 0.9, 0.9]) for h in hues]
    
    # 截取所需数量
    colors_list = colors_list[:n_classes]
    
    # 创建自定义色图
    return ListedColormap(colors_list, name=f'HighDimCMap{n_classes}')
# ====================
# 辅助函数 - 加载模型
# ====================
def load_model(path, device, input_channels=3, num_classes=10):
    """
    加载模型并处理DataParallel对象的特殊保存格式
    """
    # 加载模型数据
    model_data = torch.load(path, map_location=device, weights_only=False)
    model = torchvision_local.models.resnet34(num_classes=num_classes, method="b_spline", num_points=input_channels)
    
    # 处理DataParallel对象
    if isinstance(model_data, torch.nn.parallel.DataParallel):
        print(f"检测到DataParallel格式模型: {path}")
        state_dict = model_data.module.state_dict()
    elif hasattr(model_data, 'state_dict'):
        print(f"检测到完整模型对象: {path}")
        state_dict = model_data.state_dict()
    else:
        # 已经是状态字典
        state_dict = model_data
        print(f"加载普通状态字典: {path}")
    
    # 处理关键名称错误
    fixed_state_dict = {}
    for key, value in state_dict.items():
        # 移除可能的自定义前缀
        new_key = key.replace('torchvision_local.models.', '')
        new_key = new_key.replace('module.', '')
        fixed_state_dict[new_key] = value
    
    # # 如果需要，修改第一卷积层通道数
    # if input_channels != 3:
    #     original_conv1 = model.conv1
    #     model.conv1 = torch.nn.Conv2d(
    #         input_channels, 
    #         original_conv1.out_channels,
    #         kernel_size=original_conv1.kernel_size,
    #         stride=original_conv1.stride,
    #         padding=original_conv1.padding,
    #         bias=original_conv1.bias
    #     )
    
    # 加载修正后的状态字典
    model.load_state_dict(fixed_state_dict, strict=False)
    return model

# ====================
# 数据加载器
# ====================
def load_dataset(dataset_name, data_dir):
    """
    加载常见图像分类测试数据集 (CIFAR-10, CIFAR-100, ImageNet)
    
    参数:
        dataset_name (str): 数据集名称 ('cifar10', 'cifar100', 'imagenet')
        data_dir (str): 数据集根目录路径(ImageNet需指向包含'val'文件夹的目录)
    
    返回:
        tuple: (数据集对象, 类别名称列表, 默认输入维度)
    """
    # ImageNet验证集处理
    if dataset_name.lower() == 'imagenet':
        imagenet_val_dir = os.path.join(data_dir, 'val')
        
        # 确保验证集目录存在
        if not os.path.exists(imagenet_val_dir):
            raise FileNotFoundError(f"ImageNet验证集目录不存在，请确保目录结构为: "
                                    f"{data_dir}/val 包含1000子目录")
        
        print(f"正在加载ImageNet验证集: {imagenet_val_dir}")
        
        # ImageNet预处理
        transform = transforms.Compose([
            transforms.Resize(256),  # 将最小边缩放到256
            transforms.CenterCrop(224),  # 中心裁剪至224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 可选的批处理加载
        batch_size = 256  # 对小内存系统更友好的批大小
        
        try:
            # 从文件夹结构创建数据集
            dataset = torchvision.datasets.ImageFolder(
                root=imagenet_val_dir, transform=transform, 
            )
            
            # 创建批处理加载器
            loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=min(4, os.cpu_count()),  # 使用4个工作进程或更少
                pin_memory=True  # 如果使用GPU可提升速度
            )
            
            # 获取1000个类别名称
            class_names = dataset.classes
            
            # 如果有预定义的类别顺序则使用
            if os.path.exists(os.path.join(data_dir, 'classes.txt')):
                with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
            
            print(f"ImageNet验证集已加载: {len(dataset)} 样本, 1000类别")
            return dataset, class_names
        
        except Exception as e:
            raise RuntimeError(f"加载ImageNet数据集失败: {str(e)}")
    
    # CIFAR-10数据集处理
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)
        class_names = dataset.classes
        print("CIFAR-10测试集已加载:", len(dataset), "样本, 10类别")
        return dataset, class_names, (32, 32, 3)
    
    # CIFAR-100数据集处理
    elif dataset_name.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform)
        class_names = dataset.classes
        print("CIFAR-100测试集已加载:", len(dataset), "样本, 100类别")
        return dataset, class_names, (32, 32, 3)
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name} (请选择 'cifar10', 'cifar100' 或 'imagenet')")

# ====================
# 特征提取器
# ====================
def extract_features(model, dataloader, device, rgb):
    """从数据集中提取特征"""
    model = model.to(device).eval()
    features = []
    all_labels = []
    raw_images = []
    
    # 创建特征提取器 (移除最终分类层)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1], 
                                           torch.nn.AdaptiveAvgPool2d(1))
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            b_spline_feature
            # 获取特征 (ResNet18 的倒数第二层)
            if rgb:
                pass
            else:
                spline_feature = b_spline_feature(images, 12, 2)
                images = torch.cat((images, spline_feature), 1)
            feat = feature_extractor(images)
            feat = feat.view(feat.size(0), -1)  # 展平
            
            features.append(feat.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # 仅保存部分原始图像
            if len(raw_images) < 16:
                raw_images.append(
                    images[:min(8, images.size(0))].cpu().numpy()
                )
            
            if (i + 1) * dataloader.batch_size >= config['num_samples']:
                break
    
    features = np.vstack(features)[:config['num_samples']] if len(features) > 0 else np.array([])
    all_labels = np.concatenate(all_labels)[:config['num_samples']] if len(all_labels) > 0 else np.array([])
    
    return features, all_labels, raw_images

# ====================
# 可视化函数
# ====================
def plot_feature_space(ax, features, labels, title, class_names, raw_images=None):
    """绘制特征空间可视化图"""
    if len(features) == 0 or features.shape[1] < 2:
        ax.text(0.5, 0.5, "特征数据不足", ha='center', va='center', fontsize=12, family='Times New Roman')
        ax.set_title(title)
        return
    
    # 移除可能的NaN值
    valid_indices = ~np.isnan(features).any(axis=1)
    features = features[valid_indices]
    labels = labels[valid_indices]
    
    if len(features) < 10:
        ax.text(0.5, 0.5, "有效特征点过少", ha='center', va='center', fontsize=12, family='Times New Roman')
        ax.set_title(title)
        return
    
    # 创建颜色映射
    unique_labels = np.unique(labels)
    if config['rgb_num_classes'] <= 10:
        cmap = plt.get_cmap('tab10', config['rgb_num_classes'])
    else:
        cmap = generate_high_dim_colormap(config['rgb_num_classes'], config['palette_strategy'])
    
    # 确保类名数量匹配
    if len(class_names) < len(unique_labels):
        class_names = [f"Class {i}" for i in unique_labels]
    
    # 绘制样本点
    scatter = ax.scatter(
        features[:, 0], features[:, 1],
        c=labels, cmap=cmap, alpha=0.6,
        s=5, edgecolors='none'
    )
    
    # 计算类中心点和距离
    centroids = []
    intra_dists = []
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        if len(class_features) > 0:
            centroid = np.mean(class_features, axis=0)
            centroids.append(centroid)
            intra_dists.append(np.mean(np.linalg.norm(class_features - centroid, axis=1)))
    
    # 计算平均类内距离
    avg_intra = np.mean(intra_dists) if len(intra_dists) > 0 else 0
    
    # 计算类间距
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    avg_inter = np.mean(inter_dists) if len(inter_dists) > 0 else 0
    
    # 绘制类中心
    # 绘制类中心（带透明度调整）
    for i, centroid in enumerate(centroids):
        if not np.isnan(centroid).any():
            # 提取当前类别的颜色
            base_color = cmap(i)
            
            # 创建带有透明度的颜色
            transparent_color = (
                base_color[0],  # R
                base_color[1],  # G
                base_color[2],  # B
                0.7             # Alpha (0-1透明度)
            )
            
            # 绘制带有透明度的中心点
            ax.plot(
                centroid[0], centroid[1], 
                '*', 
                markersize=10,             # 可根据需求调整大小
                markerfacecolor=transparent_color,  # 使用透明颜色
                markeredgecolor='k',
                markeredgewidth=1,
                alpha=0.6,                 # 额外全局透明度控制（可选）
                zorder=10
            )
    
    # 设置标题和轴标签
    ax.set_title(f"{title}\nIntra: {avg_intra:.2f}, Inter: {avg_inter:.2f}", fontsize=12 , family='Times New Roman')
    ax.set_xlabel("Dimension 1", family='Times New Roman')
    ax.set_ylabel("Dimension 2", family='Times New Roman')

# ====================
# 主函数
# ====================
def visualize_feature_comparison(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    dataset, class_names = load_dataset(config['dataset_name'], config['data_dir'])
    
    # 选择子集用于可视化
    num_samples = min(config['num_samples'], len(dataset))
    print(f"使用样本数: {num_samples}")
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset_dataset = Subset(dataset, indices)
    dataloader = DataLoader(subset_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # =============================
    # 加载模型 - 解决DataParallel问题
    # =============================
    print("加载RGB模型...")
    rgb_model = load_model(config['rgb_model_path'], device, input_channels=0, num_classes=config['rgb_num_classes'])
    
    print("加载连续特征模型...")
    cont_model = load_model(config['continuous_model_path'], device, input_channels=12, num_classes=config['cont_num_classes'])
    
    # 移动到设备
    rgb_model = rgb_model.to(device).eval()
    cont_model = cont_model.to(device).eval()
    
    # =============================
    # 特征提取
    # =============================
    print("提取特征中...")
    
    # 提取RGB模型特征
    print(" > 提取RGB模型特征")
    rgb_features, rgb_labels, rgb_raw = extract_features(rgb_model, dataloader, device, rgb=True)
    
    # 提取连续特征模型特征
    print(" > 提取连续特征模型特征")
    cont_features, cont_labels, cont_raw = extract_features(cont_model, dataloader, device, rgb=False)
    
    # =============================
    # 降维处理
    # =============================
    print("降维处理中...")
    
    # 使用PCA预先降维以加速t-SNE
    pca = PCA(n_components=min(50, min(len(rgb_features), len(cont_features))), random_state=42)
    
    results = []
    dtypes = ['RGB模型', '连续特征模型']
    all_features = [rgb_features, cont_features]
    all_labels = [rgb_labels, cont_labels]
    all_raw = [rgb_raw, cont_raw]
    
    for name, features, labels, raw in zip(dtypes, all_features, all_labels, all_raw):
        if len(features) == 0:
            print(f"警告: {name} 没有提取到特征")
            results.append((np.array([]), np.array([]), raw))
            continue
            
        # PCA降维
        # n_components = min(50, features.shape[1])
        # features_pca = PCA(n_components=n_components).fit_transform(features) if n_components > 0 else features
        
        # t-SNE降维
        perp = min(config['perplexity'], max(5, len(features) // 10))
        tsne = TSNE(n_components=2, perplexity=perp, 
                   max_iter=config['n_iter'], random_state=42)
        
        print(f" > 对{name}进行t-SNE降维...")
        features_tsne = tsne.fit_transform(features)
        
        results.append((features_tsne, labels, raw))
    
    # =============================
    # 可视化
    # =============================
    print("创建可视化图形...")
    
    plt.figure(figsize=(12, 6))
    plt.suptitle("Feature Space Visualization Comparison", fontsize=16, fontweight='bold', family='Times New Roman')
    
    # RGB特征空间可视化
    ax1 = plt.subplot(1, 2, 1)
    ft1, lbl1, raw1 = results[0]
    plot_feature_space(ax1, ft1, lbl1, 
                      "RGB Feature Space", 
                      class_names, raw1)
    
    # 连续特征空间可视化
    ax2 = plt.subplot(1, 2, 2)
    ft2, lbl2, raw2 = results[1]
    plot_feature_space(ax2, ft2, lbl2, 
                      "Continuous Feature Space", 
                      class_names, raw2)
    
    # 设置通用说明
    x_label = "t-SNE Dimensionality Reduction" if perp < len(ft1) else "PCA Dimensionality Reduction"
    plt.figtext(0.5, 0.01,
    f"Dimensionality Reduction Method: {x_label} | "
    f"Points: RGB={len(ft1)}, Continuous={len(ft2)} | "
    f"Number of samples: {num_samples}",
    ha="center", fontsize=10, family='Times New Roman')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存结果
    if config['save_path']:
        plt.savefig(config['save_path'], dpi=config['dpi'], bbox_inches='tight')
        print(f"\n可视化结果已保存至: {config['save_path']}")
    
    # plt.show()
    return

# ====================
# 执行代码 - 入口点
# ====================
if __name__ == "__main__":
    visualize_feature_comparison(config)
