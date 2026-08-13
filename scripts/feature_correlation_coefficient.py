import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision import transforms

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Feature extractor class
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        
    def forward(self, x):
        return self.features(x)

# B-spline related functions
def b_spline_basis(u, knots, k, i):
    """Recursively compute B-spline basis functions"""
    if k == 0:
        condition = (knots[i] <= u) & (u < knots[i+1])
        return torch.where(condition, torch.tensor(1.0, device=u.device), torch.tensor(0.0, device=u.device))
    
    denom1 = knots[i+k] - knots[i]
    denom2 = knots[i+k+1] - knots[i+1]
    
    coeff1 = torch.zeros_like(u)
    if denom1 != 0:
        coeff1 = (u - knots[i]) / denom1
    
    coeff2 = torch.zeros_like(u)
    if denom2 != 0:
        coeff2 = (knots[i+k+1] - u) / denom2
        
    term1 = coeff1 * b_spline_basis(u, knots, k-1, i)
    term2 = coeff2 * b_spline_basis(u, knots, k-1, i+1)
    
    return term1 + term2

def b_spline_feature(control_points, num_points=128, k=2):
    """Generate B-spline curve features from control points"""
    u_values = torch.linspace(0, 1, num_points, device=control_points.device)
    n_control = control_points.size(0)
    
    # Create uniform knot vector
    knots = torch.cat([
        torch.zeros(k, device=control_points.device),
        torch.linspace(0, 1, n_control - k + 1, device=control_points.device),
        torch.ones(k, device=control_points.device)
    ])
    
    # Evaluate B-spline curve at num_points locations
    curve = torch.zeros(num_points, device=control_points.device)
    for i in range(n_control):
        curve += b_spline_basis(u_values, knots, k, i) * control_points[i]
    
    return curve

# Main analysis function
def feature_redundancy_analysis(dataset, model, sample_size=5000, pca_variance=0.95, corr_threshold=0.7):
    """
    Perform feature redundancy analysis workflow
    
    Parameters:
        dataset: ImageNet dataset
        model: Pre-trained model
        sample_size: Number of samples to analyze
        pca_variance: Variance to retain in PCA
        corr_threshold: High correlation threshold
    """
    print(f"Starting feature redundancy analysis...")
    print(f"Sample size: {sample_size}, PCA variance: {pca_variance*100:.0f}%, Corr threshold: {corr_threshold}")
    
    # Initialize model and feature extractor
    model = model.to(device)
    model.eval()
    feature_extractor = FeatureExtractor(model).to(device)
    
    # Create data loader
    indices = torch.randperm(len(dataset))[:sample_size]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True
    )
    
    # Extract features
    features = []
    print("Extracting features...")
    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device)
            
            # Apply B-spline feature transformation
            batch_features = []
            for img in images:
                # Get control points for each channel (h x w)
                ch1 = img[0].cpu()
                ch2 = img[1].cpu()
                ch3 = img[2].cpu()
                
                # Generate B-spline representation for each channel
                feature_ch1 = b_spline_feature(ch1.flatten(), num_points=128)  # Increased feature points
                feature_ch2 = b_spline_feature(ch2.flatten(), num_points=128)
                feature_ch3 = b_spline_feature(ch3.flatten(), num_points=128)
                
                # Combine features from three channels
                combined_features = torch.cat([feature_ch1, feature_ch2, feature_ch3])
                batch_features.append(combined_features.cpu().numpy())
            
            batch_features = np.array(batch_features)
            features.append(batch_features)
    
    features = np.concatenate(features, axis=0)
    print(f"Feature matrix shape: {features.shape} (samples x features)")
    print(f"Feature dimension: {features.shape[1]}, Samples: {features.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform PCA dimensionality reduction
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=pca_variance)
    pca_features = pca.fit_transform(scaled_features)
    
    print(f"Original dimensions: {scaled_features.shape[1]}")
    print(f"Dimensions after PCA: {pca_features.shape[1]}")
    retained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"Retained variance: {retained_variance:.2f}%")
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(pca_features.T)  # Transpose to compute inter-channel correlations
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        square=True,
        vmin=-1, vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(f"Channel Correlation Heatmap after PCA\n({pca_variance*100:.0f}% Variance Retained, {pca_features.shape[1]} Principal Components)", 
              fontsize=15, family='Times New Roman')
    plt.xlabel("Feature Channel Index")
    plt.ylabel("Feature Channel Index")
    
    # Mark highly correlated channel pairs
    n_components = len(correlation_matrix)
    high_corr_count = 0
    
    # Print matrix info for debugging
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    
    for i in range(n_components):
        for j in range(i+1, n_components):
            corr_val = correlation_matrix[i, j]
            if abs(corr_val) > corr_threshold:
                plt.text(j + 0.5, i + 0.5, "★", 
                        ha='center', va='center', 
                        color='white' if corr_val > 0 else 'black', 
                        fontsize=9, fontweight='bold', family='Times New Roman')
                high_corr_count += 1
    print(f"Found {high_corr_count} highly correlated pairs")
    
    # Add legend
    plt.text(n_components * 0.9, n_components * 1.05, f"★: |Correlation| > {corr_threshold}", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8), family='Times New Roman')
    
    plt.tight_layout()
    plt.savefig("imagenet_feature_correlation.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Analysis report
    total_edges = n_components * (n_components - 1) // 2
    redundancy_ratio = high_corr_count / total_edges if total_edges > 0 else 0
    
    print("\nFeature Redundancy Analysis Report:")
    print(f"Total feature channels: {features.shape[1]}")
    print(f"Channels after PCA: {pca_features.shape[1]}")
    print(f"Highly correlated channel pairs: {high_corr_count} (|r| > {corr_threshold})")
    print(f"Redundancy ratio: {redundancy_ratio*100:.2f}%")
    print(f"Possible channel pairs: {total_edges}")
    
    return correlation_matrix, pca_features

# Main function
if __name__ == "__main__":
    # 1. Load ImageNet dataset
    data_path = "/home/Datasets/imagenet"  # Change to your ImageNet path
    size = 224
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation set
    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, "val"),
        transform=preprocess
    )
    print(f"Dataset loaded. Total images: {len(dataset)}")
    
    # Load pre-trained model (ResNet50 example)
    print("Loading pre-trained ResNet50 model...")
    model = torchvision.models.resnet50(pretrained=True)
    
    # Run analysis (sample 1000 images from ImageNet)
    corr_matrix, pca_features = feature_redundancy_analysis(
        dataset, model, 
        sample_size=1000
    )
    
    # Optional: save results
    np.save("imagenet_pca_features.npy", pca_features)
    np.save("imagenet_correlation_matrix.npy", corr_matrix)
    print("Results saved successfully")
