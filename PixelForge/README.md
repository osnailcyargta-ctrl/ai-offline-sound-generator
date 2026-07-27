# PixelForge

Offline, on-device 16x16 pixel-art generator for Android. You draw your own
training examples, sort them into categories, pick how many epochs to train
for, then type a prompt and the model *synthesizes* a brand new 16x16 image
pixel by pixel — it does not copy, crop, or blend your drawings together.

Pure Kotlin, no server, no PyTorch/TensorFlow, no ML runtime. Same
from-scratch approach as the audio sibling project (SoundForge): a small GRU,
hand-written forward pass, backprop-through-time, and Adam optimizer.

## How it actually generates an image

1. **Draw** a 16x16 example on the canvas using the fixed 24-color palette,
   save it into a category (`fish`, `boom`, `hero`, ...) with optional aliases.
   A fixed palette (instead of full RGB) keeps pixel art crisp and keeps the
   model's output layer small — 24-way classification per pixel instead of
   blurry continuous color regression.
2. **TRAIN AI** runs a GRU that predicts each pixel *in raster order*
   (left-to-right, top-to-bottom), conditioned on: the pixel to its left, the
   pixel directly above it, its (row, col) position, and a learned embedding
   for the category. Position + left/above context is what lets a 1D
   sequence model learn 2D shapes with very little training data. You choose
   the epoch count; more epochs fit small/simple categories better but risk
   memorizing near-exact copies if you only drew one or two examples per
   category — a handful of varied drawings per category generalizes better
   than epoch count alone.
3. **Generating** samples pixels one at a time from the model's own predicted
   probability distribution (not argmax), feeding each choice back in as
   context for the next pixel — genuinely autoregressive synthesis. The
   Variation slider controls sampling temperature.
4. A prompt matching **one** category generates straight from it. A prompt
   matching **multiple** categories (`buatin ikan boom`) blends their learned
   embeddings into a single vector and runs *one* generation pass from that
   blend — a real model-level concept mix, not a pixel-by-pixel cross-fade
   (averaging two finished sprites just looks like ghosting, so we don't do
   that).

## Honest limitations

- With only a few hand-drawn examples per category, generated sprites will
  often closely resemble your training drawings rather than wildly novel
  compositions — that's the expected behavior of a small model with a small
  dataset, not a bug. More varied examples per category = more variety out.
- Category blending quality depends on how compatible the two learned
  concepts are; blending very different sprites can produce incoherent
  results. That's an inherent characteristic of embedding-level blending in
  a small model.

## Memory

The whole model is roughly 40-70K parameters (a few hundred KB). Training
buffers for a single 16x16 image are 256 steps of small vectors — trivial.
The app comfortably stays well under the 100MB budget; a bare Android
Activity + view framework typically accounts for most of that baseline.

## Data packs

Export writes a `.pxpack.zip` with `library.json`, the n-gram matcher, **the
trained GRU weights** (`model/pixelgru.bin`), training metadata, and every
drawing (`images/*.pix`, 256 raw palette-index bytes each). Import on another
phone, choose Merge or Replace — the trained model moves with it.

## Build

```bash
./gradlew assembleDebug
# app/build/outputs/apk/debug/app-debug.apk
```

Or push to `main` on GitHub and grab the APK from the Actions run artifacts.

Requirements: JDK 17, Android SDK 34, minSdk 24.

## Project layout

```
app/src/main/java/com/osnail/pixelforge/
  MainActivity.kt      UI, drawing, training progress, wiring
  Library.kt            categories, drawings, JSON persistence
  Trainer.kt             background GRU training loop, custom epoch count
  PixelGru.kt            from-scratch GRU: raster-order pixel prediction, BPTT, Adam
  Generator.kt            single-category or blended-embedding generation
  NgramModel.kt           char n-gram tf-idf prompt -> category matcher
  Palette.kt              fixed 24-color palette
  PixelImage.kt            16x16 palette-indexed image, bitmap conversion, binary IO
  PixelCanvasView.kt       custom View: editable drawing / read-only preview
  DataPack.kt              .pxpack export/import
```
