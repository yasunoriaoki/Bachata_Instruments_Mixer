#!/usr/bin/env python3
import argparse
import os
import queue
import threading
import time
import wave

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "sounddevice is required. Install with: pip install sounddevice"
    ) from exc

try:
    import soundfile as sf  # type: ignore
except ImportError:
    sf = None


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


def list_stems(stems_dir):
    return sorted(
        f for f in os.listdir(stems_dir) if f.lower().endswith(".wav")
    )


def load_stems(stems_dir):
    stems = list_stems(stems_dir)
    if not stems:
        raise SystemExit(f"No .wav stems found in {stems_dir}")
    audio_map = {}
    samplerate = None
    channels = None
    max_len = 0
    for stem in stems:
        path = os.path.join(stems_dir, stem)
        audio, sr = read_audio(path)
        if samplerate is None:
            samplerate = sr
            channels = audio.shape[1]
        elif sr != samplerate:
            raise SystemExit(
                f"Sample rate mismatch in {path} (expected {samplerate}, got {sr})"
            )
        if audio.shape[1] != channels:
            raise SystemExit(
                f"Channel mismatch in {path} ({audio.shape[1]}ch vs {channels}ch)"
            )
        max_len = max(max_len, audio.shape[0])
        name = os.path.splitext(stem)[0]
        audio_map[name] = audio
    for name, audio in list(audio_map.items()):
        if audio.shape[0] < max_len:
            pad = max_len - audio.shape[0]
            audio_map[name] = np.pad(audio, ((0, pad), (0, 0)), mode="constant")
    return audio_map, samplerate, channels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live remix stems with real-time gain control."
    )
    parser.add_argument(
        "--stems-dir",
        required=True,
        help="Directory containing .wav stems.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=2048,
        help="Audio block size for playback.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_map, samplerate, channels = load_stems(args.stems_dir)
    stem_names = sorted(audio_map.keys())
    gains = {name: 1.0 for name in stem_names}

    control_queue = queue.Queue()
    stop_flag = threading.Event()

    def input_thread():
        print("Type: <stem>=<gain> (e.g., drums=0.8). Type 'list' or 'quit'.")
        while not stop_flag.is_set():
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                stop_flag.set()
                break
            if line.lower() == "list":
                print("Stems:", ", ".join(stem_names))
                continue
            if "=" not in line:
                print("Use <stem>=<gain>, e.g. vocals=0.7")
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name not in gains:
                print(f"Unknown stem '{name}'. Type 'list' for stems.")
                continue
            try:
                gain = float(value)
            except ValueError:
                print("Gain must be a number.")
                continue
            control_queue.put((name, gain))

    thread = threading.Thread(target=input_thread, daemon=True)
    thread.start()

    index = 0
    total_len = max(audio.shape[0] for audio in audio_map.values())

    def callback(outdata, frames, time_info, status):
        nonlocal index
        while True:
            try:
                name, gain = control_queue.get_nowait()
                gains[name] = gain
            except queue.Empty:
                break
        end = index + frames
        if end > total_len:
            end = total_len
        mix = np.zeros((frames, channels), dtype=np.float32)
        for name, audio in audio_map.items():
            segment = audio[index:end]
            if segment.shape[0] < frames:
                segment = np.pad(segment, ((0, frames - segment.shape[0]), (0, 0)))
            mix += segment * gains[name]
        outdata[:] = np.clip(mix, -1.0, 1.0)
        index += frames
        if index >= total_len:
            stop_flag.set()
            raise sd.CallbackStop()

    print(f"Playing {total_len / samplerate:.1f}s at {samplerate} Hz.")
    print("Initial gains are 1.0 for all stems.")
    with sd.OutputStream(
        samplerate=samplerate,
        channels=channels,
        blocksize=args.blocksize,
        dtype="float32",
        callback=callback,
    ):
        while not stop_flag.is_set():
            time.sleep(0.1)

    print("Playback finished.")


if __name__ == "__main__":
    main()
