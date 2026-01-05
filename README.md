# Bachata Instruments Mixer (Cloud Deploy)

This folder contains deployment essentials for the Streamlit web UI.

## Files
- `requirements.txt`: Python dependencies for cloud deployment.
- `packages.txt`: System packages for Streamlit Cloud (installs `ffmpeg`).

## Run locally
```bash
pip install -r Bachata_Instruments_Mixer/requirements.txt
streamlit run streamlit_app.py
```

## Notes
- Upload mode works in cloud deployments.
- Local recording uses `sounddevice`, which is optional and not recommended for cloud.
- Demucs is heavy (Torch dependency) and may require more memory/time on smaller cloud instances.

## Streamlit Cloud
Place `requirements.txt` and `packages.txt` at the repo root (or configure Streamlit Cloud to use the `Bachata_Instruments_Mixer` folder) so `ffmpeg` gets installed.

## Optional: enable local recording
Uncomment `sounddevice` in `requirements.txt` and install it in a local environment.
