# SoundForge

Offline, on-device sound effect generator for Android. You upload your own sound
effects, sort them into categories, hit **TRAIN AI** to train a small generative
model right on the phone, then type a prompt like `make a boom laser sound` and
the model *synthesizes* a brand new sound from what it learned — it does not
splice, pitch-shift, or otherwise replay your recordings.

Pure Kotlin. No server, no internet, no PyTorch/TensorFlow, no third-party ML
runtime — the FFT, mel-spectrogram, vocoder, and the GRU itself (forward pass,
backprop-through-time, Adam optimizer) are all hand-written.

## How it actually generates sound

1. **Categories** — you create them (`boom`, `laser`, `fish`, ...) with aliases
   (`ledakan, explosion, blast`) so typos and synonyms still match at prompt time.
2. **Sounds** — any audio file (wav/mp3/ogg/m4a/flac) gets decoded, silence-trimmed,
   and converted to a **log-mel spectrogram** (a compact time/frequency
   representation, ~86 frames/sec instead of 44100 raw samples/sec).
3. **TRAIN AI** runs a small GRU (~97K parameters, ~760KB) that learns to predict
   the *next* mel frame given the previous one and a learned embedding for the
   sound's category — the same from-scratch approach as earlier text models,
   just regressing continuous spectral vectors instead of classifying tokens.
   Training runs on a background coroutine with a live progress bar; roughly
   2-8 minutes depending on how much audio you've added. Retraining reuses
   existing weights if the category set hasn't changed.
4. **Generating** feeds the model's own output back into itself frame by frame
   (autoregressive sampling) for as many frames as needed, with noise scaled by
   the Variation slider so nothing repeats identically — the spectral content
   is something the network invented, not a recording.
5. **Griffin-Lim** (iterative phase reconstruction, plain FFT math — no second
   network) turns that generated spectrogram back into a playable waveform.
6. A prompt matching multiple categories (`buatin ikan boom`) gives each
   category its own independent generation pass, then the resulting
   AI-synthesized audio streams are mixed down — like layering two
   synthesizers, except both layers are AI output, not samples.

## Honest limitations

- Small model + a handful of short training sounds per category means output
  leans lo-fi/textured rather than studio-clean. More sounds and more training
  time generally sharpen it.
- Griffin-Lim isn't perfect phase recovery, so there's a characteristic
  slightly "airy" quality to the reconstructed audio — a known vocoding
  tradeoff, not a bug.

## Data packs

Export writes a `.sfpack.zip` containing `library.json`, the n-gram matcher,
**the trained GRU weights** (`model/melgru.bin`), training metadata, and all
your WAV files. Copy it to another phone, hit Import, choose Merge or Replace
— the whole trained model moves with it, no retraining needed on the new device.

## Build

**Android Studio:** open the folder, let it sync, Run.

**Command line:**

```bash
./gradlew assembleDebug
# app/build/outputs/apk/debug/app-debug.apk
```

**GitHub Actions:** every push to any branch runs `.github/workflows/build.yml`.
The debug APK lands in the run's artifacts (`SoundForge-debug-apk`) and is also
attached to a `build-<run number>` release for installing straight from a phone.

Requirements: JDK 17, Android SDK 34, minSdk 24.

## Project layout

```
app/src/main/java/com/osnail/soundforge/
  MainActivity.kt    UI, pickers, wiring, training progress
  Library.kt         categories, samples, JSON persistence
  Trainer.kt         background GRU training loop + progress callback
  MelGru.kt          from-scratch GRU: forward, BPTT, Adam, generate, save/load
  MelFeatures.kt     STFT/ISTFT, mel filterbank, Griffin-Lim vocoder
  Fft.kt             radix-2 Cooley-Tukey FFT
  Generator.kt       per-category AI generation + mixdown
  NgramModel.kt      tokenizer, char n-gram tf-idf prompt matcher
  AudioDecoder.kt    MediaExtractor/MediaCodec -> mono float PCM
  WavIO.kt           WAV read/write
  Player.kt          AudioTrack playback
  DataPack.kt        .sfpack export/import (sounds + library + trained model)
```
