#!/usr/bin/env python3
import streamlit as st
import streamlit.components.v1 as components


def main():
    st.set_page_config(page_title="Bachata WebAudio Mixer", page_icon="🎧")
    st.title("Bachata WebAudio Mixer")
    st.caption("Upload stem WAVs and mix live in your browser.")

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
  <style>
    .mixer { font-family: Arial, sans-serif; }
    .row { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
    .name { width: 140px; }
    .status { color: #444; font-size: 12px; }
    button { margin-right: 8px; }
  </style>
</head>
<body>
  <div class="mixer">
    <input id="files" type="file" accept=".wav" multiple />
    <div class="row">
      <button id="play">Play</button>
      <button id="stop">Stop</button>
      <span id="status" class="status">No stems loaded.</span>
    </div>
    <div id="tracks"></div>
  </div>
  <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const fileInput = document.getElementById('files');
    const tracksDiv = document.getElementById('tracks');
    const status = document.getElementById('status');
    const playBtn = document.getElementById('play');
    const stopBtn = document.getElementById('stop');
    let buffers = [];
    let gains = [];
    let sources = [];

    function reset() {
      buffers = [];
      gains = [];
      sources = [];
      tracksDiv.innerHTML = '';
      status.textContent = 'No stems loaded.';
    }

    fileInput.addEventListener('change', async () => {
      reset();
      const files = Array.from(fileInput.files || []);
      if (!files.length) return;
      status.textContent = 'Loading...';
      for (const file of files) {
        const arrayBuffer = await file.arrayBuffer();
        const buffer = await ctx.decodeAudioData(arrayBuffer);
        buffers.push({ name: file.name.replace(/\\.[^/.]+$/, ''), buffer });
      }
      buffers.sort((a, b) => a.name.localeCompare(b.name));
      for (const item of buffers) {
        const gain = ctx.createGain();
        gain.gain.value = 1.0;
        gains.push({ name: item.name, node: gain });
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = `
          <div class="name">${item.name}</div>
          <input type="range" min="0" max="2" step="0.01" value="1" />
          <span class="value">1.00</span>
        `;
        const slider = row.querySelector('input');
        const value = row.querySelector('.value');
        slider.addEventListener('input', () => {
          const v = parseFloat(slider.value);
          value.textContent = v.toFixed(2);
          gain.gain.value = v;
        });
        tracksDiv.appendChild(row);
      }
      status.textContent = `${buffers.length} stems loaded.`;
    });

    playBtn.addEventListener('click', async () => {
      if (!buffers.length) return;
      if (ctx.state === 'suspended') await ctx.resume();
      sources.forEach(s => { try { s.stop(); } catch (e) {} });
      sources = [];
      for (let i = 0; i < buffers.length; i++) {
        const src = ctx.createBufferSource();
        src.buffer = buffers[i].buffer;
        const gainNode = gains[i].node;
        src.connect(gainNode).connect(ctx.destination);
        src.start();
        sources.push(src);
      }
      status.textContent = 'Playing...';
    });

    stopBtn.addEventListener('click', () => {
      sources.forEach(s => { try { s.stop(); } catch (e) {} });
      sources = [];
      status.textContent = 'Stopped.';
    });
  </script>
</body>
</html>
        """,
        height=380,
    )


if __name__ == "__main__":
    main()
