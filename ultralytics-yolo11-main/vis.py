# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms

#
# # 定义RCA模块
# class RCA(nn.Module):
#     def __init__(self, inp, kernel_size=1, ratio=2, band_kernel_size=11,
#                  dw_size=(1, 1), padding=(0, 0), stride=1, square_kernel_size=3, relu=True):
#         super(RCA, self).__init__()
#         self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size,
#                                    padding=square_kernel_size // 2, groups=inp)
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         gc = inp // ratio
#         self.excite = nn.Sequential(
#             nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size),
#                       padding=(0, band_kernel_size // 2), groups=gc),
#             nn.BatchNorm2d(gc),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1),
#                       padding=(band_kernel_size // 2, 0), groups=gc),
#             nn.Sigmoid()
#         )
#
#     def sge(self, x):
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x)
#         x_gather = x_h + x_w
#         ge = self.excite(x_gather)
#         return ge
#
#     def forward(self, x):
#         loc = self.dwconv_hw(x)
#         att = self.sge(x)
#         out = att * loc
#         return out, loc, att
#
#
# # 可视化函数
# def visualize_rca_process(image_path, save_path=None):
#     # 设备设置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 图像预处理
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     # 加载图像
#     image = Image.open(image_path).convert('RGB')
#     input_tensor = transform(image).unsqueeze(0).to(device)
#
#     # 初始化RCA模块
#     rca = RCA(inp=3).to(device)
#     rca.eval()
#
#     # 前向传播
#     with torch.no_grad():
#         output, loc, att = rca(input_tensor)
#
#     # 反归一化用于可视化
#     def denormalize(tensor):
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#         return tensor * std + mean
#
#     # 准备可视化数据
#     input_img = denormalize(input_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)
#     loc_img = denormalize(loc).squeeze(0).cpu().numpy().transpose(1, 2, 0)
#     att_img = att.squeeze(0).cpu().numpy().transpose(1, 2, 0)
#     output_img = denormalize(output).squeeze(0).cpu().numpy().transpose(1, 2, 0)
#
#     # 确保像素值在[0,1]范围内
#     input_img = np.clip(input_img, 0, 1)
#     loc_img = np.clip(loc_img, 0, 1)
#     output_img = np.clip(output_img, 0, 1)
#
#     # 创建可视化
#     plt.figure(figsize=(16, 12))
#
#     # 原始输入图像
#     plt.subplot(2, 2, 1)
#     plt.imshow(input_img)
#     plt.title('Input Image', fontsize=14)
#     plt.axis('off')
#
#     # 局部特征图
#     plt.subplot(2, 2, 2)
#     plt.imshow(loc_img)
#     plt.title('Local Features (after depthwise conv)', fontsize=14)
#     plt.axis('off')
#
#     # 注意力图
#     plt.subplot(2, 2, 3)
#     plt.imshow(att_img[:, :, 0], cmap='jet')  # 取第一个通道显示
#     plt.title('Attention Map', fontsize=14)
#     plt.axis('off')
#     plt.colorbar(fraction=0.046, pad=0.04)
#
#     # 最终输出
#     plt.subplot(2, 2, 4)
#     plt.imshow(output_img)
#     plt.title('Output (Attention × Local Features)', fontsize=14)
#     plt.axis('off')
#
#     plt.tight_layout()
#
#     # 保存或显示结果
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Visualization saved to {save_path}")
#     else:
#         plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 替换为您的图片路径
#     image_path = r"F:\zfm\ultralytics-yolo11-main\dataset\trans10k\images\test\77.jpg"  # 请替换为实际图片路径
#     visualize_rca_process(image_path, save_path="rca_visualization.png")
import torch
import torch.nn as nn
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


class FFEMS_Visualizer(nn.Module):
    """
    FFEMS模块可视化类
    用于展示频域分支的关键处理步骤
    """

    def __init__(self, in_channels=3, reduction=16, num_scales=3):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.num_scales = num_scales
        self.mid_channels = max(1, in_channels // reduction)

        # 频域处理相关层
        self.cutoff_params = nn.ParameterList([
            nn.Parameter(torch.tensor(0.2 * (i + 1)), requires_grad=False)
            for i in range(num_scales)
        ])

        self.freq_fusion = nn.Conv2d(
            in_channels * num_scales,
            self.mid_channels,
            kernel_size=1
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.mid_channels, max(1, self.mid_channels // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, self.mid_channels // 4), self.mid_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(self.mid_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.freq_mapping = nn.Conv2d(
            self.mid_channels,
            in_channels,
            kernel_size=1
        )

        self.layer_norm = nn.LayerNorm(in_channels)

    def forward_with_visualization(self, x):
        """
        带可视化的前向传播
        返回关键步骤的特征图
        """
        B, C, H, W = x.shape
        visualization_data = {}

        # ==================== 多尺度频域处理 ====================
        multi_scale_outputs = []
        freq_domain = fft.rfft2(x, norm='ortho')

        # 存储每个尺度的频域掩码
        freq_masks = []

        for i in range(self.num_scales):
            # 生成频率坐标网格
            y_coords = torch.linspace(-0.5, 0.5, H, device=x.device)
            x_coords = torch.linspace(-0.5, 0.5, W // 2 + 1, device=x.device)
            Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
            freq_distance = torch.sqrt(X ** 2 + Y ** 2)

            # 应用高通滤波
            cutoff = self.cutoff_params[i].clamp(0.05, 0.45)
            high_pass_mask = (freq_distance > cutoff).float()
            freq_masks.append(high_pass_mask.cpu().detach().numpy())

            # 滤波并转换回空间域
            filtered_freq = freq_domain * high_pass_mask.unsqueeze(0).unsqueeze(0)
            spatial_output = fft.irfft2(filtered_freq, s=(H, W), norm='ortho')

            # 层归一化
            spatial_output = spatial_output.permute(0, 2, 3, 1)
            spatial_output = self.layer_norm(spatial_output)
            spatial_output = spatial_output.permute(0, 3, 1, 2)

            multi_scale_outputs.append(spatial_output)

        visualization_data['freq_masks'] = freq_masks
        visualization_data['multi_scale_outputs'] = [out.cpu().detach() for out in multi_scale_outputs]

        # ==================== 频域特征融合 ====================
        fused_freq = torch.cat(multi_scale_outputs, dim=1)
        compressed_freq = self.freq_fusion(fused_freq)

        visualization_data['compressed_freq'] = compressed_freq.cpu().detach()

        # ==================== 注意力机制 ====================
        channel_weights = self.channel_attn(compressed_freq)
        weighted_freq = compressed_freq * channel_weights

        spatial_weights = self.spatial_attn(weighted_freq)
        weighted_freq = weighted_freq * spatial_weights

        visualization_data['channel_weights'] = channel_weights.cpu().detach()
        visualization_data['spatial_weights'] = spatial_weights.cpu().detach()
        visualization_data['weighted_freq'] = weighted_freq.cpu().detach()

        # ==================== 频域注意力图 ====================
        freq_attn = self.freq_mapping(weighted_freq)
        freq_attn = torch.sigmoid(freq_attn)

        visualization_data['freq_attn'] = freq_attn.cpu().detach()

        return visualization_data


def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """加载并预处理图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')

    # 预处理
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return image_tensor, image


def visualize_ffems_processing(image_path):
    """主可视化函数"""

    # 加载图像
    input_tensor, original_image = load_and_preprocess_image(image_path)

    # 创建可视化器
    visualizer = FFEMS_Visualizer(in_channels=3, reduction=16, num_scales=3)

    # 获取可视化数据
    vis_data = visualizer.forward_with_visualization(input_tensor)

    # 创建可视化图形
    fig = plt.figure(figsize=(20, 15))

    # 1. 原始图像
    plt.subplot(3, 4, 1)
    plt.imshow(np.array(original_image))
    plt.title('Original Image')
    plt.axis('off')

    # 2. 频域掩码可视化
    for i, mask in enumerate(vis_data['freq_masks']):
        plt.subplot(3, 4, i + 2)
        plt.imshow(mask, cmap='hot', extent=[-0.5, 0.5, -0.5, 0.5])
        plt.title(f'Freq Mask Scale {i + 1}\nCutoff: {visualizer.cutoff_params[i].item():.3f}')
        plt.colorbar()

    # 3. 多尺度输出可视化 (取第一个通道)
    for i, scale_output in enumerate(vis_data['multi_scale_outputs']):
        plt.subplot(3, 4, i + 5)
        output_img = scale_output[0, 0].numpy()  # 取batch=0, channel=0
        output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
        plt.imshow(output_img, cmap='gray')
        plt.title(f'Scale {i + 1} Output')
        plt.axis('off')

    # 4. compressed_freq 可视化 (取前几个通道)
    compressed_freq = vis_data['compressed_freq']
    plt.subplot(3, 4, 8)
    # 取所有通道的平均
    compressed_avg = compressed_freq[0].mean(dim=0).numpy()
    compressed_avg = (compressed_avg - compressed_avg.min()) / (compressed_avg.max() - compressed_avg.min())
    plt.imshow(compressed_avg, cmap='viridis')
    plt.title('Compressed Freq\n(Channel Average)')
    plt.axis('off')
    plt.colorbar()

    # 5. 注意力权重可视化
    plt.subplot(3, 4, 9)
    channel_weights = vis_data['channel_weights'][0].mean(dim=0).numpy()
    channel_weights = (channel_weights - channel_weights.min()) / (channel_weights.max() - channel_weights.min())
    plt.imshow(channel_weights, cmap='plasma')
    plt.title('Channel Attention Weights')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(3, 4, 10)
    spatial_weights = vis_data['spatial_weights'][0, 0].numpy()
    spatial_weights = (spatial_weights - spatial_weights.min()) / (spatial_weights.max() - spatial_weights.min())
    plt.imshow(spatial_weights, cmap='plasma')
    plt.title('Spatial Attention Weights')
    plt.axis('off')
    plt.colorbar()

    # 6. weighted_freq 可视化
    plt.subplot(3, 4, 11)
    weighted_freq = vis_data['weighted_freq'][0].mean(dim=0).numpy()
    weighted_freq = (weighted_freq - weighted_freq.min()) / (weighted_freq.max() - weighted_freq.min())
    plt.imshow(weighted_freq, cmap='viridis')
    plt.title('Weighted Freq\n(After Attention)')
    plt.axis('off')
    plt.colorbar()

    # 7. 最终频域注意力图
    plt.subplot(3, 4, 12)
    freq_attn = vis_data['freq_attn'][0].mean(dim=0).numpy()
    freq_attn = (freq_attn - freq_attn.min()) / (freq_attn.max() - freq_attn.min())
    plt.imshow(freq_attn, cmap='hot')
    plt.title('Final Freq Attention Map')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("\n=== FFEMS Processing Statistics ===")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Compressed freq shape: {compressed_freq.shape}")
    print(f"Final attention map shape: {vis_data['freq_attn'].shape}")
    print(f"Channel weights range: [{vis_data['channel_weights'].min():.3f}, {vis_data['channel_weights'].max():.3f}]")
    print(f"Spatial weights range: [{vis_data['spatial_weights'].min():.3f}, {vis_data['spatial_weights'].max():.3f}]")

    return vis_data


def visualize_individual_channels(vis_data, n_channels=4):
    """可视化单个通道的特征图"""

    compressed_freq = vis_data['compressed_freq']
    weighted_freq = vis_data['weighted_freq']
    freq_attn = vis_data['freq_attn']

    fig, axes = plt.subplots(3, n_channels, figsize=(15, 10))

    # 压缩后的频域特征通道
    for i in range(min(n_channels, compressed_freq.shape[1])):
        channel_data = compressed_freq[0, i].numpy()
        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
        axes[0, i].imshow(channel_data, cmap='viridis')
        axes[0, i].set_title(f'Compressed Freq\nChannel {i}')
        axes[0, i].axis('off')

    # 加权后的频域特征通道
    for i in range(min(n_channels, weighted_freq.shape[1])):
        channel_data = weighted_freq[0, i].numpy()
        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
        axes[1, i].imshow(channel_data, cmap='viridis')
        axes[1, i].set_title(f'Weighted Freq\nChannel {i}')
        axes[1, i].axis('off')

    # 频域注意力图通道
    for i in range(min(n_channels, freq_attn.shape[1])):
        channel_data = freq_attn[0, i].numpy()
        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
        axes[2, i].imshow(channel_data, cmap='hot')
        axes[2, i].set_title(f'Freq Attention\nChannel {i}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = r"F:\zfm\trans10k\test\hard\images\99.jpg"  # 或者使用测试图像

    # 如果没有图像，可以创建一个测试图像
    if image_path == "your_image.jpg":
        # 创建测试图像
        test_image = np.random.rand(256, 256, 3) * 255
        test_image = Image.fromarray(test_image.astype('uint8'))
        test_image.save("test_image.jpg")
        image_path = "test_image.jpg"
        print("Created test image for visualization")

    # 运行可视化
    vis_data = visualize_ffems_processing(image_path)

    # 可视化单个通道
    visualize_individual_channels(vis_data, n_channels=4)