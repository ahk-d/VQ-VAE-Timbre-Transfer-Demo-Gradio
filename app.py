import os
import tempfile
import requests
import gradio as gr
import torch
import librosa
import numpy as np
import subprocess
import sys

def install_dependencies():
    """Install required packages for deployment"""
    try:
        if not os.path.exists('ss-vq-vae'):
            print("Cloning ss-vq-vae repository...")
            subprocess.run(['git', 'clone', 'https://github.com/cifkao/ss-vq-vae.git'], check=True)
        
        subprocess.run([sys.executable, '-m', 'pip', 'install', './ss-vq-vae/src'], check=True)
        print("Dependencies installed successfully!")
        
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        raise

# Install dependencies for deployment
try:
    install_dependencies()
    import confugue
    from ss_vq_vae.models.vqvae_oneshot import Experiment
except ImportError:
    print("ss-vq-vae not found. Please install manually or run in Colab.")
    sys.exit(1)

def download_model():
    """Download model files if they don't exist"""
    model_dir = 'ss-vq-vae/experiments/model'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model_state.pt')
    if not os.path.exists(model_path):
        print("Downloading model...")
        url = 'https://adasp.telecom-paris.fr/rc-ext/demos_companion-pages/vqvae_examples/ssvqvae_model_state.pt'
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")

# Initialize model
download_model()
logdir = 'ss-vq-vae/experiments/model'
cfg = confugue.Configuration.from_yaml_file(os.path.join(logdir, 'config.yaml'))
exp = cfg.configure(Experiment, logdir=logdir, device='cpu')
exp.model.load_state_dict(torch.load(os.path.join(logdir, 'model_state.pt'), map_location=exp.device))
exp.model.train(False)

# Preset audio URLs
INPUT_ROOT = 'https://adasp.telecom-paris.fr/rc-ext/demos_companion-pages/vqvae_examples/'
INPUT_URLS = {
    'Electric Guitar': INPUT_ROOT + 'real/content/UnicornRodeo_Maybe_UnicornRodeo_Maybe_Full_25_ElecGtr2CloseMic3.0148.mp3',
    'Electric Organ': INPUT_ROOT + 'real/style/AllenStone_Naturally_Allen%20Stone_Naturally_Keys-Organ-Active%20DI.0253.mp3',
    'Jazz Piano': INPUT_ROOT + 'real/style/MaurizioPagnuttiSextet_AllTheGinIsGone_MaurizioPagnuttiSextet_AllTheGinIsGone_Full_12_PianoMics1.08.mp3',
    'Synth': INPUT_ROOT + 'real/content/Skelpolu_TogetherAlone_Skelpolu_TogetherAlone_Full_13_Synth.0190.mp3',
    'Rhodes DI': INPUT_ROOT + 'real/content/Diesel13_ColourMeRed_Diesel13_ColourMeRed_Full_30_RhodesDI.0062.mp3',
    'Acoustic Guitar Lead': INPUT_ROOT + 'real/style/NikolaStajicFtVlasisKostas_Nalim_Nikola%20Stajic%20ft.%20Vlasis%20Kostas_Nalim_Acoustic%20Guitar-Lead-Ela%20M%20251.0170.mp3',
    'Bass Amp': INPUT_ROOT + 'real/content/HurrayForTheRiffRaff_LivingInTheCity_Hurray%20for%20the%20Riff%20Raff_Livin%20in%20the%20City_Bass-Amp-M82.0018.mp3',
    'Bass Bip': INPUT_ROOT + 'real/style/RememberDecember_CUNextTime_RememberDecember_CUNextTime_Full_11_Bass_bip.041.mp3',
    'SynthFX': INPUT_ROOT + 'real/content/MR0902_JamesElder_MR0902_JamesElder_Full_13_SynthFX1.163.mp3',
    'Electric Guitar Close': INPUT_ROOT + 'real/style/Fergessen_TheWind_Fergessen_TheWind_Full_17_SlecGtr3a_Close.146.mp3',
    'Rhodes NBATG': INPUT_ROOT + 'real/content/NickiBluhmAndTheGramblers_GoGoGo_NBATG%20-%20Rhodes%20-%20DI.098.mp3',
    'Keys DI Grace': INPUT_ROOT + 'real/style/JessicaChildress_SlowDown_SD%20KEYS-DI-GRACE.147.mp3',
    'Dulcimer': INPUT_ROOT + 'real/content/ButterflyEffect_PreachRightHere_ButterflyEffect_PreachRightHere_Full_16_Dulcimer2.076.mp3',
    'Strings Section': INPUT_ROOT + 'real/style/AngeloBoltini_ThisTown_AngeloBoltini_ThisTown_Full_47_Strings_SectionMic_Vln2.0181.mp3',
    'Mellotron': INPUT_ROOT + 'real/content/Triviul_Dorothy_Triviul_Dorothy_Full_07_Mellotron.120.mp3',
    'Acoustic Guitar CU': INPUT_ROOT + 'real/style/UncleDad_WhoIAm_legend-strings_AC%20GUITAR-3-CU29-SHADOWHILL.R.0106.mp3',
    'Fiddle': INPUT_ROOT + 'real/content/EndaReilly_CurAnLongAgSeol_EndaReilly_CurAnLongAgSeol_Full_10_Fiddle2.0163.mp3',
    'Violins': INPUT_ROOT + 'real/style/ScottElliott_AeternumVale_ScottElliott_AeternumVale_Full_41_Violins.0138.mp3',
    'Upright Bass': INPUT_ROOT + 'real/content/AbletonesBigBand_SongOfIndia_UPRIGHT%20BASS%20-%20ELA%20M%20260%20-%20Neve%2033102.136.mp3',
    'Taiko': INPUT_ROOT + 'real/style/CarlosGonzalez_APlaceForUs_CarlosGonzalez_APlaceForUs_Full_21_Taiko.0115.mp3',
    'Guitar 2': INPUT_ROOT + 'real/content/AllHandsLost_Ambitions_AllHandsLost_Ambitions_Full_Guitar%202.0292.mp3',
    'Alto Sax': INPUT_ROOT + 'real/style/SunshineGarciaBand_ForIAmTheMoon_zip5-outro-uke-shaker_OUTRO%20ALTO-251E-SSL6000E.0290.mp3',
    'Bass Close Mic': INPUT_ROOT + 'real/content/DonCamilloChoir_MarshMarigoldsSong_DonCamilloChoir_MarshMarigoldsSong_Full_08_BassCloseMic2.000.mp3',
    'Electric Guitar Distorted': INPUT_ROOT + 'real/style/EnterTheHaggis_TwoBareHands_25.%20Jubilee%20Riots%20-%202%20Bar%20Hands_ELE%20Guitars-Ignater-M81.160.mp3',
    'Bells': INPUT_ROOT + 'real/content/cryonicPAX_Melancholy_cryonicPAX_Melancholy_Full_10_Bells.0034.mp3',
    'Bass Mic 647': INPUT_ROOT + 'real/style/KungFu_JoyRide_40.%20Kung%20Fu%20-%20Joy%20ride_Bass-Mic-647.0090.mp3',
}

# Separate content and style options based on URL paths
CONTENT_OPTIONS = [key for key in INPUT_URLS.keys() if any(word in INPUT_URLS[key] for word in ['content'])]
STYLE_OPTIONS = [key for key in INPUT_URLS.keys() if any(word in INPUT_URLS[key] for word in ['style'])]

# Add remaining items to both lists if they don't contain 'content' or 'style'
for key in INPUT_URLS.keys():
    if key not in CONTENT_OPTIONS and key not in STYLE_OPTIONS:
        CONTENT_OPTIONS.append(key)
        STYLE_OPTIONS.append(key)

def load_audio_from_url(url, sr=None):
    """Load audio from URL by downloading to temporary file"""
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    audio, _ = librosa.load(tmp_file_path, sr=sr)
    os.unlink(tmp_file_path)
    return audio

def preview_content_preset(preset_name):
    """Load and return audio for content preset preview"""
    if preset_name and preset_name in INPUT_URLS:
        try:
            audio = load_audio_from_url(INPUT_URLS[preset_name], sr=exp.sr)
            # Limit to 5 seconds for preview
            preview_duration = 5
            max_samples = int(preview_duration * exp.sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            return (exp.sr, audio)
        except Exception as e:
            print(f"Error loading content preset: {e}")
            return None
    return None

def preview_style_preset(preset_name):
    """Load and return audio for style preset preview"""
    if preset_name and preset_name in INPUT_URLS:
        try:
            audio = load_audio_from_url(INPUT_URLS[preset_name], sr=exp.sr)
            # Limit to 5 seconds for preview
            preview_duration = 5
            max_samples = int(preview_duration * exp.sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            return (exp.sr, audio)
        except Exception as e:
            print(f"Error loading style preset: {e}")
            return None
    return None

def process_timbre_transfer(content_file, content_preset, style_file, style_preset, max_duration=8):
    """Process timbre transfer with uploaded files or presets"""
    try:
        # Load content audio (musical notes/melody to preserve)
        if content_file is not None:
            a_content, _ = librosa.load(content_file, sr=exp.sr)
        else:
            if content_preset and content_preset in INPUT_URLS:
                a_content = load_audio_from_url(INPUT_URLS[content_preset], sr=exp.sr)
            else:
                return None, "Please upload a content file or select a content preset"

        # Load style audio (timbre/texture to apply)
        if style_file is not None:
            a_style, _ = librosa.load(style_file, sr=exp.sr)
        else:
            if style_preset and style_preset in INPUT_URLS:
                a_style = load_audio_from_url(INPUT_URLS[style_preset], sr=exp.sr)
            else:
                return None, "Please upload a style file or select a style preset"

        # Limit duration to prevent memory issues
        max_samples = int(max_duration * exp.sr)
        if len(a_content) > max_samples:
            a_content = a_content[:max_samples]
        if len(a_style) > max_samples:
            a_style = a_style[:max_samples]

        # Preprocess: Convert audio to model input format
        s_content = torch.as_tensor(exp.preprocess(a_content), device=exp.device)[None, :]
        s_style = torch.as_tensor(exp.preprocess(a_style), device=exp.device)[None, :]
        l_content, l_style = (torch.as_tensor([x.shape[2]], device=exp.device) for x in [s_content, s_style])

        # Run model: Extract content features, extract style features, then recombine
        with torch.no_grad():
            s_output = exp.model(input_c=s_content, input_s=s_style,
                               length_c=l_content, length_s=l_style)

        # Postprocess: Convert model output back to audio waveform
        a_output = exp.postprocess(s_output.cpu().numpy()[0])

        return (exp.sr, a_output), "Transfer completed successfully!"

    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="VQ-VAE Timbre Transfer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 VQ-VAE Timbre Transfer Demo
    """)

    # Two-column layout for content and style inputs
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎼 Content Audio")
            content_file = gr.Audio(label="Upload Content Audio", type="filepath")
            content_preset = gr.Dropdown(
                choices=[""] + CONTENT_OPTIONS,
                label="Or choose preset",
                value=""
            )
            # Preview audio for content preset
            content_preview = gr.Audio(
                label="🔊 Content Preview (5s)",
                interactive=False,
                visible=False
            )

        with gr.Column():
            gr.Markdown("### 🎨 Style Audio")
            style_file = gr.Audio(label="Upload Style Audio", type="filepath")
            style_preset = gr.Dropdown(
                choices=[""] + STYLE_OPTIONS,
                label="Or choose preset",
                value="Electric Guitar Close"
            )
            # Preview audio for style preset
            style_preview = gr.Audio(
                label="🔊 Style Preview (5s)",
                interactive=False,
                visible=True  # Visible by default since we have a default selection
            )

    # Duration control to balance quality vs processing time
    max_duration = gr.Slider(1, 15, value=8, step=1, label="Max Duration (seconds)")

    process_btn = gr.Button("🚀 Transfer Timbre", variant="primary", size="lg")

    # Output section
    with gr.Row():
        output_audio = gr.Audio(label="🎵 Output Audio", interactive=False)
        status_msg = gr.Textbox(label="Status", interactive=False, max_lines=3)

    # Hide previews when user uploads their own files
    content_file.change(
        fn=lambda file: gr.update(visible=False) if file is not None else None,
        inputs=[content_file],
        outputs=[content_preview]
    )

    style_file.change(
        fn=lambda file: gr.update(visible=False) if file is not None else None,
        inputs=[style_file],
        outputs=[style_preview]
    )

    # Connect preset selection to audio preview (only when no file uploaded)
    content_preset.change(
        fn=lambda preset, file: (
            preview_content_preset(preset) if preset and file is None else None,
            gr.update(visible=bool(preset and file is None))
        ),
        inputs=[content_preset, content_file],
        outputs=[content_preview, content_preview]
    )

    style_preset.change(
        fn=lambda preset, file: (
            preview_style_preset(preset) if preset and file is None else None,
            gr.update(visible=bool(preset and file is None))
        ),
        inputs=[style_preset, style_file],
        outputs=[style_preview, style_preview]
    )

    # Load default style preview on startup
    demo.load(
        fn=lambda: preview_style_preset("Electric Guitar Close"),
        outputs=[style_preview]
    )

    # Connect button click to processing function
    process_btn.click(
        fn=process_timbre_transfer,
        inputs=[content_file, content_preset, style_file, style_preset, max_duration],
        outputs=[output_audio, status_msg]
    )

    gr.Markdown("""
    ### 🔧 Troubleshooting
    - **Poor transfer quality?** Try different instrument combinations or adjust max duration
    - **Audio doesn't load?** Check internet connection or try different presets
    - **Processing slow?** Reduce max duration or try shorter audio clips
    
    ### 📖 Citation
    Original work by Ondřej Cífka (InterDigital R&D and Télécom Paris, 2020).  
    Demo by Ali Dulaimi.
    """)

if __name__ == "__main__":
    demo.launch(share=True, debug=True, height=1400)
