package com.osnail.soundforge

import java.util.Locale
import kotlin.math.max
import kotlin.random.Random

class GenResult(val pcm: FloatArray, val log: String)

object Generator {

    /**
     * Builds a new sound from every matched category.
     * Each layer gets a random pitch / decay / filter / delay so no two
     * generations sound identical, then everything is mixed down.
     */
    fun generate(
        lib: Library,
        matches: List<Match>,
        variation: Float,
        seed: Long = System.nanoTime()
    ): GenResult {
        val rnd = Random(seed)
        val layers = ArrayList<FloatArray>()
        val gains = ArrayList<Float>()
        val offsets = ArrayList<Int>()
        val log = StringBuilder()

        val v = variation.coerceIn(0f, 1f)

        for (m in matches) {
            val cat = lib.find(m.category) ?: continue
            if (cat.samples.isEmpty()) {
                log.append("• ${cat.name}: no samples, skipped\n")
                continue
            }
            val sample = cat.samples[rnd.nextInt(cat.samples.size)]
            var pcm = lib.loadSample(sample)
            if (pcm.isEmpty()) {
                log.append("• ${cat.name}: sample file missing, skipped\n")
                continue
            }
            pcm = pcm.copyOf()

            val pitch = (1f + (rnd.nextFloat() * 2f - 1f) * (0.35f * v)).coerceIn(0.55f, 1.9f)
            pcm = Dsp.resample(pcm, pitch)

            val decay = rnd.nextFloat() * 0.6f * v
            Dsp.applyDecay(pcm, decay)

            val lp = rnd.nextFloat() * 0.5f * v
            Dsp.lowpass(pcm, lp)

            Dsp.addNoise(pcm, 0.15f * v * rnd.nextFloat(), rnd)
            Dsp.normalize(pcm, 0.95f)
            Dsp.fade(pcm)

            // The first layer that actually made it anchors the mix at t=0 —
            // keying off the match index would leave dead air whenever an
            // earlier category got skipped.
            val delayMs = if (layers.isEmpty()) 0 else (rnd.nextFloat() * 90f * v).toInt()
            val offset = delayMs * WavIO.SAMPLE_RATE / 1000

            layers.add(pcm)
            offsets.add(offset)

            log.append(
                "• ${cat.name}  (match ${(m.score * 100).toInt()}%, pitch ${fmt(pitch)}x, +${delayMs}ms)\n"
            )
        }

        if (layers.isEmpty()) return GenResult(FloatArray(0), "No usable layers.")

        // Headroom scales with the layers that survived, not the ones we matched.
        val gain = (0.95f / max(1f, layers.size * 0.72f)).coerceAtLeast(0.35f)
        repeat(layers.size) { gains.add(gain) }

        var out = Dsp.mix(layers, gains, offsets)
        out = Dsp.normalize(out, 0.9f)
        out = Dsp.fade(out, 2, 18)

        val secs = out.size.toFloat() / WavIO.SAMPLE_RATE
        log.append("Rendered ${layers.size} layer(s), ${fmt(secs)}s")
        return GenResult(out, log.toString())
    }

    /** Locale-independent so the log never reads "1,25x" on an id-ID phone. */
    private fun fmt(v: Float): String = String.format(Locale.US, "%.2f", v)
}
