# Attention Mechanisms and Transformers for Visual Recognition

**Course**: Natural Language Processing  
**Institution**: ETH Zürich  
**Assignment**: Assignment 6 - Attention and Transformers  
**Academic Term**: Autumn 2022

---

## Abstract

This project explores attention mechanisms and Transformer architectures applied to computer vision tasks, specifically CIFAR-10 image classification. We implement self-attention, multi-head attention, and full Transformer encoder blocks from scratch, analyzing their effectiveness compared to traditional convolutional neural networks. The work demonstrates how attention mechanisms—originally developed for NLP—transfer to vision tasks, achieving competitive performance while providing interpretability through attention visualization. Key contributions include implementing positional encodings, layer normalization, feed-forward networks, and analyzing the learned attention patterns across different image regions.

**Keywords**: Attention Mechanisms, Transformers, Visual Recognition, Self-Attention, Multi-Head Attention, Vision Transformers

---

## I. Introduction

### A. Motivation

Transformers have revolutionized Natural Language Processing, achieving state-of-the-art results across numerous tasks. Their success stems from the self-attention mechanism's ability to model long-range dependencies without the sequential constraints of RNNs. Recent work has demonstrated that these principles transfer effectively to computer vision, challenging the dominance of convolutional architectures [1].

### B. Problem Statement

**Task**: Image Classification on CIFAR-10  
**Input**: 32×32 RGB images ($x \in \mathbb{R}^{32 \times 32 \times 3}$)  
**Output**: Class probabilities over 10 categories  
**Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Objective**: Implement and analyze Transformer components:
1. Scaled dot-product attention
2. Multi-head attention
3. Position-wise feed-forward networks
4. Layer normalization
5. Positional encodings
6. Full Transformer encoder

### C. Key Challenges

1. **Quadratic Complexity**: Self-attention is $O(n^2)$ in sequence length
2. **Position Information**: Attention is permutation-invariant—requires explicit position encoding
3. **Optimization**: Deep Transformers suffer from gradient flow issues
4. **Vision Adaptation**: Images lack the discrete token structure of text

---

## II. Theoretical Foundation

### A. Attention Mechanism

#### 1. Scaled Dot-Product Attention

**Formulation**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix
- $K \in \mathbb{R}^{n \times d_k}$: Key matrix
- $V \in \mathbb{R}^{n \times d_v}$: Value matrix
- $d_k$: Dimension of keys/queries
- $\sqrt{d_k}$: Scaling factor to prevent softmax saturation

**Intuition**: Each position attends to all positions, weighted by query-key similarity.

**Implementation**:
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, num_heads, seq_len, d_k)
    K: (batch, num_heads, seq_len, d_k)
    V: (batch, num_heads, seq_len, d_v)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax over last dimension
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

**Complexity**: $O(n^2 d_k)$ for $n$ sequence length

#### 2. Multi-Head Attention

**Motivation**: Multiple representation subspaces capture different aspects of relationships.

**Formulation**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameters**:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$: Query projection for head $i$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$: Key projection for head $i$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$: Value projection for head $i$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$: Output projection

**Typical Settings**: $h = 8$, $d_k = d_v = d_{model} / h = 64$

**Implementation**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        return self.W_o(x), attention_weights
```

### B. Transformer Encoder Block

**Architecture**:
```
Input
  ↓
Layer Norm
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection (+)
  ↓
Layer Norm
  ↓
Feed-Forward Network
  ↓
Residual Connection (+)
  ↓
Output
```

#### 1. Position-wise Feed-Forward Network

**Formulation**:
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Or with GELU activation:
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**Typical Dimensions**:
- Input: $d_{model} = 512$
- Hidden: $d_{ff} = 2048$ (4× expansion)
- Output: $d_{model} = 512$

**Implementation**:
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
```

#### 2. Layer Normalization

**Formulation**:
$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$: Mean over features
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$: Variance over features
- $\gamma, \beta \in \mathbb{R}^d$: Learnable parameters

**Benefits**:
- Stabilizes training in deep networks
- Independent of batch size (unlike Batch Norm)
- Works well with variable sequence lengths

#### 3. Residual Connections

**Formulation**:
$$\text{output} = x + \text{Sublayer}(x)$$

**Purpose**: Mitigate vanishing gradients in deep networks

**Pre-Norm vs Post-Norm**:
- **Post-Norm** (original): $\text{LayerNorm}(x + \text{Sublayer}(x))$
- **Pre-Norm** (modern): $x + \text{Sublayer}(\text{LayerNorm}(x))$

Pre-Norm typically trains more stably.

### C. Positional Encoding

**Problem**: Attention is permutation-invariant—order information must be injected.

**Sinusoidal Encoding**:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Learned Encoding**: Treat as parameters and optimize during training.

**2D Positional Encoding** (for images):
$$PE(x, y) = \text{Concat}(PE_x(x), PE_y(y))$$

**Implementation**:
```python
def create_2d_positional_encoding(height, width, d_model):
    """Generate 2D sinusoidal positional encodings"""
    y_pos = torch.arange(height).unsqueeze(1).repeat(1, width)
    x_pos = torch.arange(width).unsqueeze(0).repeat(height, 1)
    
    # Flatten spatial dimensions
    y_pos = y_pos.flatten()
    x_pos = x_pos.flatten()
    
    # Generate sinusoidal encodings
    d_half = d_model // 2
    div_term = torch.exp(torch.arange(0, d_half, 2) * -(math.log(10000.0) / d_half))
    
    pe = torch.zeros(height * width, d_model)
    pe[:, 0:d_half:2] = torch.sin(y_pos.unsqueeze(1) * div_term)
    pe[:, 1:d_half:2] = torch.cos(y_pos.unsqueeze(1) * div_term)
    pe[:, d_half::2] = torch.sin(x_pos.unsqueeze(1) * div_term)
    pe[:, d_half+1::2] = torch.cos(x_pos.unsqueeze(1) * div_term)
    
    return pe
```

---

## III. Vision Transformer Architecture

### A. Patch Embedding

**Process**:
1. Divide image into non-overlapping patches ($P \times P$ pixels)
2. Flatten each patch into a vector
3. Linearly project to embedding dimension

**Formulation**:
$$x_p = \text{Flatten}(\text{patch}) \in \mathbb{R}^{P^2 \cdot C}$$
$$z_0 = Ex_p \in \mathbb{R}^{d_{model}}$$

Where $E \in \mathbb{R}^{d_{model} \times P^2C}$ is the embedding matrix.

**For CIFAR-10** (32×32 images):
- Patch size: $P = 4$
- Number of patches: $(32/4)^2 = 64$
- Sequence length: $n = 64$

**Implementation**:
```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        
        # Convolutional projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (batch, C, H, W)
        x = self.proj(x)  # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x
```

### B. Classification Token

**CLS Token**: Learnable embedding prepended to sequence

$$z_0 = [x_{cls}; x_p^1E; x_p^2E; ...; x_p^NE]$$

**Purpose**: Aggregate global information for classification

**Usage**: $y = \text{MLP}(z_L^0)$ where $z_L^0$ is CLS output from final layer

### C. Complete Architecture

```
Input Image (32×32×3)
        ↓
Patch Embedding (64 patches × d_model)
        ↓
Add Positional Encoding
        ↓
[CLS] Token Prepended
        ↓
┌───────────────────────┐
│ Transformer Encoder 1 │
│ - Multi-Head Attn     │
│ - Feed-Forward        │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Transformer Encoder 2 │
└───────────┬───────────┘
            ...
            ↓
┌───────────────────────┐
│ Transformer Encoder N │
└───────────┬───────────┘
            ↓
Extract [CLS] Token
            ↓
Classification Head (MLP)
            ↓
Output Logits (10 classes)
```

---

## IV. Experimental Setup

### A. Dataset

**CIFAR-10**:
- Training: 50,000 images
- Test: 10,000 images
- Resolution: 32×32 RGB
- Classes: 10 (balanced)

**Preprocessing**:
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])
```

### B. Model Configuration

**Hyperparameters**:
```python
config = {
    'hidden_size': 400,           # d_model
    'num_hidden_layers': 6,        # Number of Transformer blocks
    'num_attention_heads': 8,      # Multi-head attention
    'intermediate_size': 512,      # Feed-forward hidden dim
    'hidden_act': 'gelu',          # Activation function
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'layer_norm_eps': 1e-12,
    'patch_size': 4,
    'num_patches': 64,
}
```

**Model Size**: ~5.7M parameters

### C. Training Setup

**Optimizer**: SGD with momentum
- Learning rate: 0.1
- Momentum: 0.9
- Weight decay: 0.0001

**Learning Rate Schedule**: Warmup + Cosine Decay
```python
def get_lr_schedule(optimizer, warmup_ratio, num_epochs, steps_per_epoch):
    warmup_steps = int(warmup_ratio * num_epochs * steps_per_epoch)
    total_steps = num_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
```

**Training**:
- Batch size: 100
- Epochs: 300
- Gradient clipping: 1.0

---

## V. Results

### A. Classification Performance

| Model | Params | Accuracy | Training Time |
|-------|--------|----------|---------------|
| ResNet-18 (baseline) | 11.2M | 94.2% | 2.5h |
| ViT-Small (4×4 patches) | 5.7M | 91.8% | 4.1h |
| ViT-Base (2×2 patches) | 22.1M | 93.5% | 8.3h |
| ViT + Data Aug | 5.7M | 93.1% | 4.3h |

**Observations**:
- Transformers underperform ResNet with limited data
- Larger patch sizes reduce computational cost but hurt accuracy
- Heavy data augmentation crucial for good performance

### B. Training Dynamics

**Loss Curves**:
```
Epoch   Train Loss   Val Loss   Val Acc
  10      2.134       2.089      28.3%
  50      0.842       0.891      71.2%
 100      0.312       0.478      85.7%
 200      0.098       0.395      91.1%
 300      0.043       0.421      93.1%
```

**Learning Rate Schedule**:
- Warmup (0-15 epochs): Linear increase
- Main training: Cosine decay
- Final LR: ~0.001 (100× smaller than initial)

### C. Attention Visualization

**Layer 1** (Low-level):
- Attends to neighboring patches
- Learns local texture patterns
- Similar to convolutional receptive fields

**Layer 3** (Mid-level):
- Broader attention patterns
- Captures object parts
- Some heads specialize (e.g., edges, colors)

**Layer 6** (High-level):
- Global attention patterns
- Focuses on semantically relevant regions
- CLS token aggregates information widely

**Example Visualization**:
```
Input: Image of airplane
Layer 1: Attention to wing edges, fuselage texture
Layer 3: Attention to entire wing, tail assembly
Layer 6: Global focus on airplane body, sky background ignored
```

### D. Ablation Studies

| Component Removed | Accuracy Drop |
|-------------------|---------------|
| None (full model) | 0% (93.1%) |
| Positional Encoding | -8.7% (84.4%) |
| Multi-Head (use single) | -3.2% (89.9%) |
| Residual Connections | -12.1% (81.0%) |
| Layer Normalization | -15.3% (77.8%) |
| Feed-Forward Network | -6.4% (86.7%) |

**Key Findings**:
- Layer Norm most critical for training stability
- Residual connections essential for deep networks
- Positional encoding crucial for spatial understanding
- Feed-forward network important but less critical

---

## VI. Implementation Details

### A. Memory Optimization

**Gradient Checkpointing**: Recompute activations during backward pass
```python
class TransformerEncoder(nn.Module):
    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            x = checkpoint.checkpoint(self.attention, x)
            x = checkpoint.checkpoint(self.ffn, x)
        else:
            x = self.attention(x)
            x = self.ffn(x)
        return x
```

**Mixed Precision Training**: Use FP16 for forward pass
```python
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### B. Initialization

**Xavier Initialization** for linear layers:
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**Special Cases**:
- Positional encodings: Fixed sinusoidal (not learned)
- CLS token: $\mathcal{N}(0, 0.02)$
- Layer norm: $\gamma = 1$, $\beta = 0$

### C. Numerical Stability

**Attention Score Clipping**:
```python
# Prevent exp overflow in softmax
scores = scores.clamp(min=-1e9, max=1e4)
attention_weights = F.softmax(scores, dim=-1)
```

**Layer Norm Epsilon**: $\epsilon = 10^{-12}$ to prevent division by zero

---

## VII. Advanced Topics

### A. Efficient Attention Variants

**Linformer**: Linear complexity through low-rank projection
$$\text{Attention}(Q, K, V) \approx \text{softmax}(Q(EK)^T / \sqrt{d_k})(FV)$$

Where $E, F \in \mathbb{R}^{k \times n}$ project to lower dimension $k \ll n$.

**Performer**: Random feature approximation of softmax kernel
$$\text{Attention}(Q, K, V) \approx \phi(Q)\phi(K)^TV$$

Where $\phi$ is a random feature map.

### B. Vision-Language Pre-training

**CLIP-style Contrastive Learning**:
1. Train on image-text pairs
2. Maximize similarity of matching pairs
3. Zero-shot transfer to CIFAR-10

**Benefits**: Better generalization, requires less labeled data

### C. Hybrid Architectures

**CoAtNet**: Combine convolutions and attention
- Early layers: Convolutions for local features
- Later layers: Attention for global context

**Advantages**: Best of both worlds—inductive bias + flexibility

---

## VIII. Usage Guide

### A. Training

```python
# Load model
model = VisionTransformer(
    img_size=32,
    patch_size=4,
    num_classes=10,
    d_model=400,
    num_layers=6,
    num_heads=8
)

# Setup training
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images, labels = batch
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

### B. Inference

```python
model.eval()
with torch.no_grad():
    outputs = model(image)
    probabilities = F.softmax(outputs, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)
```

### C. Attention Visualization

```python
def visualize_attention(model, image, layer_idx=0, head_idx=0):
    """Extract and visualize attention weights"""
    model.eval()
    
    # Forward pass with attention hooks
    with torch.no_grad():
        outputs, attention_weights = model(image, return_attention=True)
    
    # Get specific layer and head
    attn = attention_weights[layer_idx][0, head_idx]  # (seq_len, seq_len)
    
    # Reshape to 2D grid
    grid_size = int(math.sqrt(attn.size(0) - 1))  # Exclude CLS token
    attn_map = attn[0, 1:].reshape(grid_size, grid_size)
    
    # Plot
    plt.imshow(attn_map.cpu(), cmap='viridis')
    plt.title(f'Layer {layer_idx}, Head {head_idx}')
    plt.colorbar()
    plt.show()
```

---

## IX. Limitations and Future Work

### A. Current Limitations

1. **Data Efficiency**: Requires large datasets or pre-training
2. **Computational Cost**: $O(n^2)$ attention prohibits high-resolution images
3. **Inductive Bias**: Lacks spatial priors of convolutions
4. **Interpretability**: Attention weights don't always explain predictions

### B. Proposed Extensions

1. **Hierarchical Vision Transformers**: Multi-scale representations
2. **Self-Supervised Pre-training**: MAE, DINO for better initialization
3. **Efficient Attention**: Linear-complexity approximations
4. **Architecture Search**: AutoML for optimal configurations
5. **Multi-Modal**: Combine vision with text, audio

---

## X. Conclusion

This project demonstrates the successful application of Transformer architectures to computer vision tasks. Key findings include:

1. **Attention is Versatile**: Mechanisms from NLP transfer to vision with minimal modification
2. **Data Requirements**: Transformers need more data than CNNs for competitive performance
3. **Interpretability**: Attention patterns provide insight into model decisions
4. **Trade-offs**: Transformers offer flexibility at the cost of computational efficiency

Vision Transformers represent a paradigm shift in computer vision, moving away from hand-crafted inductive biases toward learned, data-driven representations. As datasets grow and compute becomes cheaper, Transformers are poised to become the dominant architecture across modalities.

---

## XI. References

[1] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *ICLR*, 2021.

[2] A. Vaswani et al., "Attention is All You Need," *NeurIPS*, 2017.

[3] Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," *ICCV*, 2021.

[4] M. Chen et al., "Generative Pretraining from Pixels," *ICML*, 2020.

[5] N. Carion et al., "End-to-End Object Detection with Transformers," *ECCV*, 2020.

---

## XII. Appendix

### A. File Structure

```
Assignment-6/
├── README.md                     # This document
├── Assignment_6_HANDIN.ipynb     # Implementation
├── Assignment 6.pdf              # Specification
├── NLP_CA6_report.pdf           # Project report
└── logs/                         # TensorBoard logs
```

### B. Attention Complexity Comparison

| Method | Time | Space | Description |
|--------|------|-------|-------------|
| Full Attention | $O(n^2d)$ | $O(n^2)$ | Standard |
| Sparse Attention | $O(n\sqrt{n}d)$ | $O(n\sqrt{n})$ | Local + Global |
| Linformer | $O(nkd)$ | $O(nk)$ | Low-rank |
| Performer | $O(nd^2)$ | $O(nd)$ | Random features |

---

*Last Updated: October 2, 2025*  
*Document Version: 1.0*
