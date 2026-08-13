import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 列出所有可用的字体
fonts = [f.name for f in fm.fontManager.ttflist]
print("Times New Roman 已安装:", "Times New Roman" in fonts)
print(fonts)

# 查看字体的确切名称
for font in fm.findSystemFonts():
    try:
        f = fm.get_font(font)
        if 'Times New Roman' in f.family_name.lower():
            print(f"找到 Times New Roman: {font} ({f.family_name})")
    except RuntimeError:
        continue

# 强制使用 Times New Roman（如果存在）
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    
    plt.figure()
    plt.title("使用 Times New Roman", fontsize=14)
    plt.plot(range(10))
    plt.savefig('using_times_new_roman.pdf')
except RuntimeError:
    print("警告：Times New Roman 不可用，回退到默认字体")
