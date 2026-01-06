#!/usr/bin/env python3
import json
import io
import os
import subprocess
import sys
import time
import uuid
import wave
import zipfile

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    import soundfile as sf  # type: ignore
except ImportError:
    sf = None

sd = None


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "webui_sessions")
MODEL_NAME = "htdemucs"
ENABLE_SESSION_LOAD = os.environ.get("ENABLE_SESSION_LOAD", "1").lower() not in (
    "0",
    "false",
    "no",
)


def ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def read_wav_wave(path):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        samplerate = wf.getframerate()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    if sampwidth == 2:
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth * 8} bits in {path}")
    audio = audio.reshape(-1, channels)
    return audio, samplerate


def read_audio(path):
    if sf is not None:
        audio, samplerate = sf.read(path, always_2d=True, dtype="float32")
        return audio, samplerate
    return read_wav_wave(path)


def write_wav(path, audio, samplerate):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())


def audio_duration_seconds(path):
    if sf is not None:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        samplerate = wf.getframerate()
    return float(frames) / float(samplerate)


@st.cache_data(show_spinner=False)
def spectrogram_png_cached(path, title, cutoff_hz=None):
    audio, samplerate = read_audio(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.specgram(audio, Fs=samplerate, NFFT=1024, noverlap=512, cmap="magma")
    if cutoff_hz is not None:
        ax.axhline(cutoff_hz, color="white", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_demucs(audio_path, out_dir, device="cpu"):
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        MODEL_NAME,
        "--device",
        device,
        "--out",
        out_dir,
        audio_path,
    ]
    subprocess.run(cmd, check=True)


def run_demucs_with_progress(audio_path, out_dir, device, progress_cb):
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        MODEL_NAME,
        "--device",
        device,
        "--out",
        out_dir,
        audio_path,
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    start = time.time()
    while True:
        ret = process.poll()
        elapsed = time.time() - start
        progress_cb(elapsed)
        if ret is not None:
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
            return
        time.sleep(0.25)


def list_input_devices():
    global sd
    if sd is None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            return []
    devices = sd.query_devices()
    items = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] == 0:
            continue
        label = f"{idx}: {dev['name']}"
        items.append((idx, label, dev))
    return items


def record_audio_with_progress(input_device, samplerate, channels, duration, progress_cb):
    global sd
    if sd is None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sounddevice is not installed") from exc
    frames = int(duration * samplerate)
    recording = sd.rec(
        frames,
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        device=input_device,
    )
    start = time.time()
    while True:
        elapsed = time.time() - start
        progress = min(elapsed / duration, 1.0) if duration > 0 else 1.0
        progress_cb(progress, elapsed)
        if elapsed >= duration:
            break
        time.sleep(0.1)
    sd.wait()
    return recording


def list_stems(stems_dir):
    if not os.path.isdir(stems_dir):
        return []
    return sorted(f for f in os.listdir(stems_dir) if f.lower().endswith(".wav"))


def session_metadata_path(session_dir):
    return os.path.join(session_dir, "meta.json")


def write_session_metadata(session_dir, input_name, stems_dir, stems):
    payload = {
        "input_name": input_name,
        "stems_dir": stems_dir,
        "stems": stems,
        "created_at": time.time(),
    }
    with open(session_metadata_path(session_dir), "w") as fh:
        json.dump(payload, fh)


def list_sessions():
    if not os.path.isdir(SESSIONS_DIR):
        return []
    sessions = []
    for entry in os.listdir(SESSIONS_DIR):
        session_dir = os.path.join(SESSIONS_DIR, entry)
        if not os.path.isdir(session_dir):
            continue
        meta_path = session_metadata_path(session_dir)
        meta = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = None
        stems_dir = None
        stems = []
        input_name = "recorded.wav"
        created_at = os.path.getmtime(session_dir)
        if meta:
            stems_dir = meta.get("stems_dir")
            stems = meta.get("stems") or []
            input_name = meta.get("input_name") or input_name
            created_at = meta.get("created_at") or created_at
        else:
            model_dir = os.path.join(session_dir, MODEL_NAME)
            if os.path.isdir(model_dir):
                track_dirs = [
                    os.path.join(model_dir, d)
                    for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d))
                ]
                if track_dirs:
                    stems_dir = track_dirs[0]
                    stems = [os.path.splitext(s)[0] for s in list_stems(stems_dir)]
        if stems_dir and os.path.isdir(stems_dir):
            sessions.append(
                {
                    "id": entry,
                    "input_name": input_name,
                    "stems_dir": stems_dir,
                    "stems": stems,
                    "created_at": created_at,
                }
            )
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return sessions


def safe_extract_zip(zip_bytes, dest_dir):
    with zipfile.ZipFile(zip_bytes) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if name.startswith("/") or ".." in name.split("/"):
                raise ValueError("Invalid zip entry path.")
        zf.extractall(dest_dir)


def zip_stems_dir(stems_dir):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(stems_dir):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, stems_dir)
                zf.write(full_path, arcname)
    mem.seek(0)
    return mem


def mix_stems(stems_dir, gains):
    stems = list_stems(stems_dir)
    if not stems:
        raise RuntimeError("No stems found.")

    mixed = None
    samplerate = None
    for stem_file in stems:
        stem_name = os.path.splitext(stem_file)[0]
        gain = float(gains.get(stem_name, 1.0))
        path = os.path.join(stems_dir, stem_file)
        audio, sr = read_audio(path)
        if samplerate is None:
            samplerate = sr
        elif sr != samplerate:
            raise RuntimeError(
                f"Sample rate mismatch in {path} (expected {samplerate}, got {sr})"
            )
        if mixed is None:
            mixed = np.zeros_like(audio)
        if audio.shape[0] > mixed.shape[0]:
            pad = audio.shape[0] - mixed.shape[0]
            mixed = np.pad(mixed, ((0, pad), (0, 0)), mode="constant")
        elif audio.shape[0] < mixed.shape[0]:
            pad = mixed.shape[0] - audio.shape[0]
            audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant")
        if audio.shape[1] != mixed.shape[1]:
            raise RuntimeError(
                f"Channel mismatch in {path} ({audio.shape[1]}ch vs {mixed.shape[1]}ch)"
            )
        mixed += audio * gain

    return mixed, samplerate, stems


def split_audio_low_high(audio, samplerate, cutoff_hz):
    if audio.ndim == 1:
        audio = audio[:, None]
    n = audio.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / samplerate)
    low_mask = freqs < cutoff_hz
    low = np.zeros_like(audio)
    high = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        spec = np.fft.rfft(audio[:, ch])
        low[:, ch] = np.fft.irfft(spec * low_mask, n=n)
        high[:, ch] = np.fft.irfft(spec * (~low_mask), n=n)
    return low, high


def mix_stems_with_split(stems_dir, gains, split_drums, cutoff_hz):
    stems = list_stems(stems_dir)
    if not stems:
        raise RuntimeError("No stems found.")

    mixed = None
    samplerate = None
    for stem_file in stems:
        stem_name = os.path.splitext(stem_file)[0]
        path = os.path.join(stems_dir, stem_file)
        audio, sr = read_audio(path)
        if samplerate is None:
            samplerate = sr
        elif sr != samplerate:
            raise RuntimeError(
                f"Sample rate mismatch in {path} (expected {samplerate}, got {sr})"
            )
        if mixed is None:
            mixed = np.zeros_like(audio)
        if audio.shape[0] > mixed.shape[0]:
            pad = audio.shape[0] - mixed.shape[0]
            mixed = np.pad(mixed, ((0, pad), (0, 0)), mode="constant")
        elif audio.shape[0] < mixed.shape[0]:
            pad = mixed.shape[0] - audio.shape[0]
            audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant")
        if audio.shape[1] != mixed.shape[1]:
            raise RuntimeError(
                f"Channel mismatch in {path} ({audio.shape[1]}ch vs {mixed.shape[1]}ch)"
            )

        if split_drums and stem_name == "drums":
            low, high = split_audio_low_high(audio, samplerate, cutoff_hz)
            mixed += low * float(gains.get("drums_low", 1.0))
            mixed += high * float(gains.get("drums_high", 1.0))
        else:
            mixed += audio * float(gains.get(stem_name, 1.0))

    return mixed, samplerate, stems


def new_session_dir(upload_name):
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    track_name = os.path.splitext(os.path.basename(upload_name))[0] or "recorded"
    stems_dir = os.path.join(session_dir, MODEL_NAME, track_name)
    return session_id, session_dir, stems_dir


def main():
    global sd
    st.set_page_config(page_title="Bachata Instruments Mixer", page_icon="🎛️")
    st.title("Bachata Instruments Mixer")
    st.caption("Upload audio, separate stems with Demucs, then remix with sliders.")

    ensure_dirs()

    if "session_id" not in st.session_state:
        st.session_state.session_id = None
        st.session_state.stems_dir = None
        st.session_state.stems = []
    if "separating" not in st.session_state:
        st.session_state.separating = False
    if "spec_images" not in st.session_state:
        st.session_state.spec_images = {}
        st.session_state.spec_meta = {}

    sessions = list_sessions()
    if ENABLE_SESSION_LOAD and sessions:
        with st.expander("Load previous separation", expanded=False):
            labels = [
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(s['created_at']))} - {s['input_name']}"
                for s in sessions
            ]
            selected = st.selectbox("Saved sessions", options=labels, index=0)
            if st.button("Load"):
                idx = labels.index(selected)
                session = sessions[idx]
                st.session_state.session_id = session["id"]
                st.session_state.stems_dir = session["stems_dir"]
                st.session_state.stems = session["stems"]
                st.success("Loaded saved separation.")

    with st.expander("Import separated stems (zip)", expanded=False):
        uploaded_zip = st.file_uploader("Upload stems zip", type=["zip"])
        if uploaded_zip is not None:
            if st.button("Import stems"):
                session_id = uuid.uuid4().hex
                session_dir = os.path.join(SESSIONS_DIR, session_id)
                stems_dir = os.path.join(session_dir, "stems")
                os.makedirs(stems_dir, exist_ok=True)
                try:
                    safe_extract_zip(uploaded_zip, stems_dir)
                except Exception as exc:  # pragma: no cover
                    st.error(f"Failed to import zip: {exc}")
                    return
                stems = [os.path.splitext(s)[0] for s in list_stems(stems_dir)]
                if not stems:
                    st.error("No .wav stems found in the zip.")
                    return
                write_session_metadata(
                    session_dir,
                    uploaded_zip.name,
                    stems_dir,
                    stems,
                )
                st.session_state.session_id = session_id
                st.session_state.stems_dir = stems_dir
                st.session_state.stems = stems
                st.success("Imported stems.")
    record_option = "Record (may not work on cloud)"
    devices_available = False
    if sd is None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            sd = None
    if sd is not None:
        devices_available = bool(list_input_devices())
    source_options = ["Upload file"]
    if devices_available:
        source_options.append(record_option)
    source = st.radio(
        "Input source",
        options=source_options,
        index=0,
        horizontal=True,
    )
    uploaded = None
    record_settings = None
    if source == "Upload file":
        uploaded = st.file_uploader(
            "Upload WAV/MP3", type=["wav", "mp3", "m4a", "flac", "ogg"]
        )
    else:
        st.info(
            "Recording uses Python sounddevice (local machine only). "
            "On cloud deployments, use file upload instead."
        )
        if sd is None:
            st.warning(
                "Recording is disabled because sounddevice is not installed. "
                "Use Upload file on cloud, or install sounddevice for local runs."
            )
        else:
            devices = list_input_devices()
            if not devices:
                st.error("No input devices found.")
            else:
                with st.expander("Available input devices", expanded=False):
                    for idx, label, dev in devices:
                        st.write(
                            f"{idx}: {dev['name']} "
                            f"({dev['max_input_channels']}ch, "
                            f"default {int(dev['default_samplerate'])} Hz)"
                        )
                device_labels = [d[1] for d in devices]
                selected = st.selectbox("Input device", options=device_labels, index=0)
                selected_idx = device_labels.index(selected)
                device_index, _label, dev = devices[selected_idx]
                default_sr = int(dev["default_samplerate"])
                max_channels = int(dev["max_input_channels"])
                channels = st.number_input(
                    "Channels", min_value=1, max_value=max_channels, value=min(2, max_channels)
                )
                samplerate = st.number_input(
                    "Sample rate (Hz)",
                    min_value=8000,
                    max_value=192000,
                    value=default_sr,
                    step=1000,
                )
                duration = st.number_input(
                    "Duration (seconds)",
                    min_value=1.0,
                    max_value=600.0,
                    value=10.0,
                    step=1.0,
                )
                recording_name = st.text_input(
                    "Recording name (optional)",
                    value="",
                    help="Used to label saved separations.",
                )
                record_settings = {
                    "device_index": device_index,
                    "channels": int(channels),
                    "samplerate": int(samplerate),
                    "duration": float(duration),
                    "recording_name": recording_name.strip() or "recorded.wav",
                }

    device = st.selectbox("Demucs device", options=["cpu", "mps", "cuda"], index=0)

    if uploaded is not None:
        st.subheader("Preview")
        st.audio(uploaded)

    separate_disabled = False
    if source == "Upload file" and uploaded is None:
        separate_disabled = True
    if source == "Record (sounddevice)" and (
        record_settings is None
    ):
        separate_disabled = True

    if st.button(
        "Separate instruments",
        type="primary",
        disabled=separate_disabled or st.session_state.separating,
    ):
        st.session_state.separating = True
        session_id, session_dir, stems_dir = new_session_dir(
            uploaded.name if uploaded else "recorded.wav"
        )
        os.makedirs(session_dir, exist_ok=True)
        if source == "Upload file":
            input_path = os.path.join(session_dir, uploaded.name)
            with open(input_path, "wb") as fh:
                fh.write(uploaded.getbuffer())
        else:
            input_path = os.path.join(session_dir, "recorded.wav")
            with st.spinner("Recording..."):
                try:
                    progress = st.progress(0.0)
                    status = st.empty()
                    def progress_cb(pct, elapsed):
                        progress.progress(pct)
                        remaining = max(record_settings["duration"] - elapsed, 0.0)
                        status.caption(
                            f"Recording... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining"
                        )
                    recording = record_audio_with_progress(
                        record_settings["device_index"],
                        record_settings["samplerate"],
                        record_settings["channels"],
                        record_settings["duration"],
                        progress_cb,
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Recording failed: {exc}")
                    return
            write_wav(input_path, recording, record_settings["samplerate"])
            st.subheader("Recorded preview")
            st.audio(input_path)
        progress = st.progress(0.0)
        status = st.empty()
        estimated = None
        try:
            estimated = audio_duration_seconds(input_path) * 3.0
        except Exception:
            estimated = None
        with st.spinner("Running Demucs..."):
            try:
                def progress_cb(elapsed):
                    if estimated:
                        pct = min(elapsed / estimated, 0.95)
                    else:
                        pct = 0.5 if elapsed > 1.0 else 0.1
                    progress.progress(pct)
                    status.caption(f"Separating... {elapsed:.1f}s elapsed")
                run_demucs_with_progress(input_path, session_dir, device=device, progress_cb=progress_cb)
                progress.progress(1.0)
                status.caption("Separation complete.")
            except subprocess.CalledProcessError as exc:
                st.error(f"Demucs failed with code {exc.returncode}.")
                return
            finally:
                st.session_state.separating = False
        stems = list_stems(stems_dir)
        if not stems:
            st.error("No stems found after separation.")
            return
        input_name = os.path.basename(input_path)
        if source != "Upload file" and record_settings is not None:
            input_name = record_settings.get("recording_name") or input_name
        write_session_metadata(
            session_dir,
            input_name,
            stems_dir,
            [os.path.splitext(s)[0] for s in stems],
        )
        st.session_state.session_id = session_id
        st.session_state.stems_dir = stems_dir
        st.session_state.stems = [os.path.splitext(s)[0] for s in stems]
        st.success("Stems ready.")

    if st.session_state.stems_dir:
        st.subheader("Remix")
        if "download_zip" not in st.session_state:
            st.session_state.download_zip = None
        if st.button("Prepare stems zip"):
            st.session_state.download_zip = zip_stems_dir(
                st.session_state.stems_dir
            )
        if st.session_state.download_zip is not None:
            st.download_button(
                "Download stems (zip)",
                data=st.session_state.download_zip,
                file_name="stems.zip",
                mime="application/zip",
            )
        split_drums = False
        cutoff_hz = 3000
        if "drums" in st.session_state.stems:
            split_drums = st.checkbox("Separate bongo and güira (heuristic)", value=True)
            if split_drums:
                cutoff_hz = st.number_input(
                    "Bongo/Güira split cutoff (Hz)",
                    min_value=500,
                    max_value=8000,
                    value=3000,
                    step=100,
                )
        gains = {}
        for stem in st.session_state.stems:
            label = stem
            if stem == "other":
                label = "guitars"
            if split_drums and stem == "drums":
                gains["drums_low"] = st.slider(
                    "bongo gain", min_value=0.0, max_value=2.0, value=1.0, step=0.05
                )
                gains["drums_high"] = st.slider(
                    "güira gain", min_value=0.0, max_value=2.0, value=1.0, step=0.05
                )
                continue
            gains[stem] = st.slider(
                f"{label} gain", min_value=0.0, max_value=2.0, value=1.0, step=0.05
            )
        show_specs = st.checkbox("Show spectrograms", value=False)
        if show_specs:
            current_meta = {
                "stems_dir": st.session_state.stems_dir,
                "cutoff_hz": cutoff_hz,
                "split_drums": split_drums,
            }
            auto_refresh = st.checkbox("Auto-refresh spectrograms on cutoff change", value=True)
            if st.session_state.spec_meta != current_meta:
                st.session_state.spec_images = {}
                st.session_state.spec_meta = current_meta
                if auto_refresh:
                    images = {}
                    for stem in st.session_state.stems:
                        stem_path = os.path.join(
                            st.session_state.stems_dir, f"{stem}.wav"
                        )
                        label = stem
                        if stem == "drums" and split_drums:
                            label = "bongo/güira"
                        if stem == "other":
                            label = "guitars"
                        cutoff = cutoff_hz if (split_drums and stem == "drums") else None
                        try:
                            images[stem] = spectrogram_png_cached(
                                stem_path,
                                f"{label} spectrogram",
                                cutoff_hz=cutoff,
                            )
                        except Exception as exc:  # pragma: no cover
                            st.error(f"Failed to load {stem}.wav: {exc}")
                            continue
                    st.session_state.spec_images = images
            if st.button("Update spectrograms"):
                images = {}
                for stem in st.session_state.stems:
                    stem_path = os.path.join(
                        st.session_state.stems_dir, f"{stem}.wav"
                    )
                    label = stem
                    if stem == "drums" and split_drums:
                        label = "bongo/güira"
                    if stem == "other":
                        label = "guitars"
                    cutoff = cutoff_hz if (split_drums and stem == "drums") else None
                    try:
                        images[stem] = spectrogram_png_cached(
                            stem_path,
                            f"{label} spectrogram",
                            cutoff_hz=cutoff,
                        )
                    except Exception as exc:  # pragma: no cover
                        st.error(f"Failed to load {stem}.wav: {exc}")
                        continue
                st.session_state.spec_images = images
            for stem in st.session_state.stems:
                stem_path = os.path.join(
                    st.session_state.stems_dir, f"{stem}.wav"
                )
                label = stem
                if stem == "drums" and split_drums:
                    label = "bongo/güira"
                if stem == "other":
                    label = "guitars"
                with st.expander(f"{label} spectrogram", expanded=False):
                    image = st.session_state.spec_images.get(stem)
                    if image is None:
                        st.caption("Click 'Update spectrograms' to render.")
                    else:
                        st.image(image)
        normalize = st.checkbox("Normalize mix", value=True)
        if st.button("Render Remix"):
            try:
                mixed, samplerate, _stems = mix_stems_with_split(
                    st.session_state.stems_dir, gains, split_drums, cutoff_hz
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return
            if normalize:
                peak = float(np.max(np.abs(mixed)))
                if peak > 0:
                    mixed = mixed / peak * 0.999
            remix_path = os.path.join(
                SESSIONS_DIR, st.session_state.session_id, "remix.wav"
            )
            write_wav(remix_path, mixed, samplerate)
            st.audio(remix_path)
            st.success("Remix ready.")


if __name__ == "__main__":
    main()
