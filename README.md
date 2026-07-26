# SoundForge

Offline, on-device sound effect generator for Android. You upload your own sound
effects, sort them into categories, train a character n-gram model right on the
phone, then type a prompt like `make a boom laser sound` and it builds a new
sound by mixing and randomizing samples from every category it detects.

No server. No internet. No ML runtime. Pure Kotlin + the Android media stack.

## How it works

1. **Categories** — you create them (`boom`, `laser`, `fish`, ...) and give each
   one aliases (`ledakan, explosion, blast`). Aliases are what make fuzzy
   matching good.
2. **Sounds** — load any audio file (wav / mp3 / ogg / m4a / flac) into a
   category. It gets decoded to mono 44.1kHz, silence-trimmed, and stored as WAV
   in app-private storage.
3. **Train** — builds a character 3-gram tf-idf index over your category names
   and aliases, saved to `model.json`. Takes milliseconds, the file is a few KB.
4. **Generate** — the prompt is tokenized, stopwords dropped, then every
   remaining word is scored against every category. Anything above the threshold
   becomes a layer. `buatin ikan boom` → `ikan` + `boom` → both layers mixed.
5. Each layer gets randomized pitch, decay, low-pass, noise and start offset
   (controlled by the Variation slider), so the same prompt never gives the exact
   same sound twice.

Fuzzy matching means typos still land: `bom` → `boom`, `lasser` → `laser`.

## Data packs

Export writes a `.sfpack.zip` containing `library.json`, `model.json` and all
your WAV files. Copy it to another phone, hit Import, choose Merge or Replace.
That's the whole sync story — no accounts, no cloud.

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

## Memory footprint

The model is a few KB. RAM at runtime is dominated by decoded audio, roughly
176 KB per second of sample held in memory during generation — with a handful of
short effects per category it sits comfortably under 100 MB. Imports are capped
at 20 seconds per file to keep it safe on low-RAM phones.

## Project layout

```
app/src/main/java/com/osnail/soundforge/
  MainActivity.kt    UI, pickers, wiring
  Library.kt         categories, samples, JSON persistence
  NgramModel.kt      tokenizer, char n-gram tf-idf matcher
  Generator.kt       layer selection + randomization + mixdown
  Dsp.kt             resample, envelope, filter, mix, normalize
  AudioDecoder.kt    MediaExtractor/MediaCodec → mono float PCM
  WavIO.kt           WAV read/write
  Player.kt          AudioTrack playback
  DataPack.kt        .sfpack export/import
```
