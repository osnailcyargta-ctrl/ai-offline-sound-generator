package com.osnail.soundforge

import kotlin.random.Random

data class GenResult(val pcm: FloatArray, val log: String)

/**
 * Every matched category gets its OWN pass through the trained GRU
 * (real autoregressive synthesis, not sample playback), vocoded with
 * Griffin-Lim, then the resulting AI-generated audio streams are mixed
 * down -- the same way a sound designer layers two synthesizers.
 */
object Generator {

    fun generate(
        model: MelGru,
        matches: List<Match>,
        variation: Float,
        secondsPerLayer: Float = 1.3f,
        seed: Long = System.nanoTime()
    ): GenResult {
        val rnd = Random(seed)
        val v = variation.coerceIn(0f, 1f)
        val temperature = 0.15 + 0.65 * v
        val framesPerLayer = (secondsPerLayer * WavIO.SAMPLE_RATE / MelFeatures.HOP).toInt().coerceAtLeast(8)

        val layers = ArrayList<FloatArray>()
        val offsets = ArrayList<Int>()
        val gains = ArrayList<Float>()
        val log = StringBuilder()

        for (m in matches) {
            if (!model.categories.contains(m.category)) {
                log.append("• ${m.category}: not in trained model, skipped\n")
                continue
            }
            val catIdx = model.categoryIndex(m.category)
            val mel = model.generate(
                catIdx,
                nFrames = framesPerLayer,
                temperature = temperature,
                seed = rnd.nextLong()
            )
            var audio = MelFeatures.melToAudio(mel, WavIO.SAMPLE_RATE, nIter = 24)
            audio = Dsp.trimSilence(audio, 0.008f)
            if (audio.size < MelFeatures.HOP) {
                log.append("• ${m.category}: generated silence, skipped\n")
                continue
            }
            Dsp.normalize(audio, 0.95f)
            Dsp.fade(audio)

            // The first layer that actually rendered anchors the mix at t=0 —
            // keying off the match index would leave dead air whenever an
            // earlier category was skipped.
            val delayMs = if (layers.isEmpty()) 0 else (rnd.nextFloat() * 90f * v).toInt()
            val offset = delayMs * WavIO.SAMPLE_RATE / 1000

            layers.add(audio)
            offsets.add(offset)
            log.append("• ${m.category} (match ${(m.score * 100).toInt()}%, AI-generated, +${delayMs}ms)\n")
        }

        if (layers.isEmpty()) return GenResult(FloatArray(0), "No usable layers.\n$log")

        // Headroom scales with the layers that rendered, not the ones we matched.
        val gain = (0.95f / maxOf(1f, layers.size * 0.72f)).coerceAtLeast(0.35f)
        repeat(layers.size) { gains.add(gain) }

        var out = Dsp.mix(layers, gains, offsets)
        out = Dsp.normalize(out, 0.9f)
        out = Dsp.fade(out, 2, 20)

        val secs = out.size.toFloat() / WavIO.SAMPLE_RATE
        log.append("Rendered ${layers.size} AI-generated layer(s), ${"%.2f".format(secs)}s")
        return GenResult(out, log.toString())
    }
}
