# Eigenfaces & PCA Face Reconstruction Studio

An interactive, web-based demonstration of **Principal Component Analysis (PCA)** and the classic **Eigenfaces** method for facial image representation, compression, and reconstruction.

Inspired by [scikit-learn's facial recognition tutorial](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html), this project demonstrates:
1. **The Power of PCA & Correlated Features**: How 1,850 raw pixel dimensions can be compressed into just 20 to 50 linear projection weights while preserving recognizable human identities.
2. **Interactive Linear Combinations**: How facial features (illumination, jaw width, eye sockets, smiles) materialize progressively from an average face ($\boldsymbol{\mu}$) as components ($K$) are added.
3. **Algorithmic Representation Bias & Geographic Skew**: How training on demographic distributions skewed toward Western male political figures directly impairs reconstruction fidelity for underrepresented global populations (the "Generalization Gap").

---

## 🌟 Key Interactive Features

### 1. Face Reconstruction Studio
- **24 Curated Global Figures**:
  - *Standard LFW Training Benchmark*: George W. Bush, Tony Blair, Colin Powell, Gerhard Schröder, Donald Rumsfeld, Hugo Chávez.
  - *Global South & Developing Nations*: Nelson Mandela, Kofi Annan, Megawati Sukarnoputri, Hu Jintao, Junichiro Koizumi, Mahathir Mohamad, Atal Bihari Vajpayee, Luiz Inácio Lula da Silva, Vicente Fox, Hamid Karzai.
  - *Women & Cultural Icons*: Serena Williams, Gloria Macapagal-Arroyo, Salma Hayek, Jackie Chan, Condoleezza Rice, Jennifer Aniston, Arnold Schwarzenegger, Bill Clinton.
- **Component Depth Slider ($K \in [0, 80]$)**:
  - Watch a face evolve in real time from the generic average face ($K=0$) to coarse contours ($K=5$), structural facial features ($K=15$), and recognizable identity ($K \ge 25$).
  - Animated emergence playback mode.
- **Trio Canvas Display**:
  - **Ground Truth Target ($\mathbf{x}$)**: Original photograph.
  - **PCA Reconstruction ($\hat{\mathbf{x}}_K$)**: Real-time linear combination.
  - **Residual Heatmap ($|\mathbf{x} - \hat{\mathbf{x}}_K|$)**: Pixel-level error map highlighting missing contours, wrinkles, or skin tone shifts.
- **Diagnostic Telemetry**:
  - Compression ratio (e.g. $92.5\times$ at $K=20$).
  - Cumulative variance explained (up to $89.1\%$ at $K=80$).
  - Mean Squared Error ($MSE$), Peak Signal-to-Noise Ratio ($PSNR$), and Cosine Similarity.
  - Bar chart of active projection weights ($w_k$).

### 2. Eigenface Feature Mixer & Synthesizer
- Manipulate individual projection weights ($w_1 \dots w_6$) directly.
- Discover what correlated features each principal component controls:
  - $\mathbf{v}_1$: Horizontal key lighting / shadow direction.
  - $\mathbf{v}_2$: Vertical ambient lighting & forehead contrast.
  - $\mathbf{v}_3$: Face elongation & jawline width.
  - $\mathbf{v}_4$: Eye socket depth & nose bridge prominence.
  - $\mathbf{v}_5$: Mouth curvature & smile expression.
  - $\mathbf{v}_6$: Hairline & upper skull contour.
- 1-click **Random Synthetic Face** generator to sample novel faces from the eigenspace manifold.

### 3. Top 16 Eigenface Gallery
- Visual cards displaying the Mean Face and the first 16 Eigenfaces.
- Click any eigenface to immediately set $K$ to include that component.

### 4. Demographic Bias & Generalization Gap Explorer
- **The Skew in Numbers**:
  - In scikit-learn's top LFW benchmark ($\ge 70$ photos), **100% are male** and **88.5% are White/Caucasian** from developed Western nations (US, UK, Germany).
  - A single individual (George W. Bush) accounts for **41.1% (530 of 1,288 images)** of the entire training set.
- **Reconstruction Error Disparity**:
  - In-distribution faces (e.g. Bush) converge rapidly ($MSE = 0.00237$ at $K=20$).
  - Out-of-distribution faces (e.g. Megawati Sukarnoputri at $MSE = 0.01372$ or Serena Williams at $MSE = 0.01084$) suffer **4.6× to 5.8× higher reconstruction error**.
  - Interactive **MSE vs $K$ Canvas Chart** illustrating the persistent generalization gap across component depths.

---

## 📐 Mathematical Formulation

Every 50×37 grayscale face image is represented as a flattened vector $\mathbf{x} \in \mathbb{R}^D$ where $D = 1850$.

Given a training set of $M = 1288$ centered images $\boldsymbol{\Phi}_i = \mathbf{x}_i - \boldsymbol{\mu}$, PCA finds an orthonormal basis of eigenvectors $\mathbf{v}_k$ by diagonalizing the covariance matrix:

$$\mathbf{C} = \frac{1}{M} \sum_{i=1}^M \boldsymbol{\Phi}_i \boldsymbol{\Phi}_i^T$$

The scalar projection weight $w_k$ of any face $\mathbf{x}$ onto the $k$-th eigenface is:

$$w_k = (\mathbf{x} - \boldsymbol{\mu}) \cdot \mathbf{v}_k$$

The reconstructed face using $K$ components is the linear combination:

$$\hat{\mathbf{x}}_K = \boldsymbol{\mu} + \sum_{k=1}^K w_k \mathbf{v}_k$$

---

## 🚀 How to Run Locally

Serve the project using any local HTTP server:

```bash
# Python 3
python3 -m http.server 8125

# Node.js
npx serve .
```

Open `http://localhost:8125` in your browser.

---

## 🌐 Deploying to GitHub Pages

1. Push this directory to your GitHub repository.
2. In your repository on GitHub, navigate to **Settings** > **Pages**.
3. Under **Build and deployment**:
   - **Source**: `GitHub Actions` (using the included `.github/workflows/pages.yml`) or `Deploy from a branch` (branch: `main`, folder: `/ (root)`).
4. Within 1–2 minutes, your demo will be live at `https://<your-username>.github.io/<repo-name>/`.
