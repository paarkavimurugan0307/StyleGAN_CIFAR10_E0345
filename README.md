# StyleGAN-Style Image Generation on CIFAR-10 (TensorFlow/Keras)

A compact, from-scratch implementation of a **StyleGAN-inspired generative adversarial network** trained on a subset of CIFAR-10. The experiment demonstrates the core ideas of StyleGAN: a mapping network, style-based generation with AdaIN, learned constant input, and per-layer noise injection. It also includes **style mixing** and **latent interpolation**.

**Notebook:** `E0324045_DL_StyleGAN.ipynb`

---

## Objective

- Understand and implement the key architectural components of StyleGAN on a small scale.
- Train a generator and discriminator adversarially on 32×32 colour images.
- Visualise how styles control image synthesis (style mixing) and how the latent space behaves (interpolation).

---

## Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 (auto-downloaded via `keras.datasets.cifar10`) |
| Images used | First 20,000 training images (deterministic subset) |
| Image shape | 32 × 32 × 3 |
| Preprocessing | Pixels scaled to **[-1, 1]** |
| Batch size | 64 (shuffled, `drop_remainder=True`) |

---

## Model Architecture

### Generator (`StyleGenerator`) – ~1.04M parameters
- **Mapping Network:** 8 fully connected layers (Z → W, dimension 128) with LeakyReLU (α=0.2). Input latent is normalised (pixel norm) first.
- **Learned Constant Input:** a trainable 4×4×128 tensor is the starting point instead of a noise vector.
- **Synthesis blocks (`StyledConv`):** 8 styled convolutions at resolutions 4 → 8 → 16 → 32 (channels 128, 128, 64, 32). Each consists of:
  - 3×3 convolution
  - **Noise injection** with a learned per-layer strength
  - **AdaIN** (layer-normalised features modulated by scale and bias predicted from `w`)
  - LeakyReLU
- **Upsampling:** nearest-neighbour resize between resolution stages.
- **toRGB:** 1×1 convolution with `tanh` output.

### Discriminator – ~428K parameters
- 4 strided 4×4 convolution blocks (32, 64, 128, 128 filters) with LeakyReLU.
- Flatten followed by a Dense layer producing a single logit.

---

## Training Setup

| Setting | Value |
|---|---|
| Framework | TensorFlow 2.21 / Keras |
| Loss | Non-saturating logistic GAN loss (softplus) |
| Optimizer | Adam (lr = 2e-4, β₁ = 0.0, β₂ = 0.99) for both G and D |
| Epochs | 10 |
| Latent dimension | 128 (Z and W) |
| Seed | 42 |
| Hardware | GPU (≈134 s total training time) |

---

## Results

### Training Losses

| Epoch | D loss | G loss |
|---|---|---|
| 1 | 1.399 | 0.716 |
| 5 | 1.390 | 0.717 |
| 10 | 1.386 | 0.693 |

The losses settle around D ≈ 1.386 (2·ln 2) and G ≈ 0.693 (ln 2). This means the discriminator is outputting near-chance predictions on real vs. fake images. With only 10 epochs, a small network and a 20K-image subset, the generated samples capture limited detail and are not photorealistic. Longer training, a larger model, regularisation (e.g. R1 penalty) and more data would be needed to improve quality.

### Visualisations (in the notebook)
1. **Real CIFAR-10 samples** – a 2×8 grid of training images.
2. **Training loss curves** – generator vs. discriminator loss per epoch.
3. **Generated samples** – 64 images from a fixed latent batch after training.
4. **Style mixing** – styles from latent A are used for early layers and styles from latent B for later layers (crossover at layers 2, 4 and 6), showing that early layers affect coarse structure and later layers affect finer appearance.
5. **Latent interpolation** – 9 images generated along a straight line between two latent vectors.

### Output Figures

**Model size**
![Model parameters](results/model_parameters.png)

**Training log (10 epochs)**
![Training log](results/training_log.png)

**Real CIFAR-10 samples**
![Real samples](results/real_samples.png)

**Training losses**
![Training losses](results/training_losses.png)

**Generated samples after training**
![Generated samples](results/generated_samples.png)

**Style mixing**
![Style mixing](results/style_mixing.png)

**Latent interpolation**
![Latent interpolation](results/latent_interpolation.png)

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/paarkavimurugan0307/StyleGAN_CIFAR10_E0345.git
cd StyleGAN_CIFAR10_E0345

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook E0324045_DL_StyleGAN.ipynb
```

Run all cells in order. CIFAR-10 is downloaded automatically on first run. A GPU is recommended but not required.

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Matplotlib

---

## Key Concepts Demonstrated

- **Mapping network** – disentangles the latent space (Z → W).
- **AdaIN-based style modulation** – injects style at every synthesis layer.
- **Learned constant input** – synthesis begins from a trainable tensor.
- **Stochastic noise injection** – adds fine, per-pixel variation.
- **Style mixing** – combines coarse and fine attributes from different latents.
- **Latent-space interpolation** – smooth transitions between generated images.

---

## Limitations and Future Work

- Short training (10 epochs) and a 20K-image subset limit sample quality.
- Simplified StyleGAN: no progressive growing, weight demodulation (StyleGAN2), equalised learning rate, or path-length regularisation.
- Possible improvements: longer training, FID/IS evaluation, R1 gradient penalty, exponential moving average of generator weights, and higher-resolution datasets.

---

## References

- Karras, Laine, Aila. *A Style-Based Generator Architecture for Generative Adversarial Networks* (StyleGAN), CVPR 2019.
- Goodfellow et al. *Generative Adversarial Nets*, NIPS 2014.
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10), 2009.

---

## Author

**Paarkavi M R** – E0324045  
B.Tech CSE (Artificial Intelligence and Data Analytics)  
Sri Ramachandra Institute of Higher Education and Research
