# 🎵 VQ-VAE Timbre Transfer Demo

Transfer the **timbre** (tone/texture) from one audio source to another while preserving the **musical content** using **Vector Quantized Variational AutoEncoders (VQ-VAE)**.

---

## 🚀 Quick Start

### 🔗 Online Demo

- **Hugging Face Space**: [ahk-d/timbre-transfer-demo](https://huggingface.co/spaces/ahk-d/VQ-VAE-Timbre-Transfer-Demo)  
- **Google Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10cUu2sW-Cu1rapX1ey1FLI2zez4AnMH3#scrollTo=jCUYFJUF3m4o)

---

### 💻 Local Setup

```bash
# Clone the repository
git clone https://github.com/ahk-d/VQ-VAE-Timbre-Transfer-Demo-Gradio.git
cd VQ-VAE-Timbre-Transfer-Demo-Gradio
```

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Run the Gradio app
python app.py
```

---

## 🎼 How It Works

This demo uses a **VQ-VAE (Vector Quantized Variational AutoEncoder)** model with three main components:

- **Content Encoder**: Extracts musical content (notes, rhythm) while discarding timbre.
- **Style Encoder**: Captures timbral characteristics from the style audio.
- **Decoder**: Reconstructs audio by combining content with the new timbre.

---

## 🎯 Why Some Combinations Work Better

- **Harmonic similarity**: Instruments with similar harmonic structures transfer more effectively.
- **Spectral compatibility**: Overlapping frequency ranges produce cleaner results.
- **Temporal characteristics**: Similar attack/decay patterns preserve musical expression.

---

## 📁 Usage

### 🎵 Content vs Style

- **Content Audio**: The musical notes/melody you want to preserve.
- **Style Audio**: The instrument timbre/texture you want to apply.

---

### 💡 Tips for Best Results

- Use **clear, non-noisy** audio clips.
- **8-second clips** provide a good balance between quality and processing time.
- **Harmonic instruments** (like piano or guitar) often transfer well to each other.
- Try various combinations — sometimes unexpected pairs can produce musically interesting results!

---
