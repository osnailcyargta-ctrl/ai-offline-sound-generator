package com.osnail.soundforge

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

object Dsp {

    /** Linear-interpolation resample. step > 1 = shorter/higher pitch. */
    fun resample(input: FloatArray, step: Float): FloatArray {
        if (input.isEmpty() || step <= 0f) return input
        val outLen = (input.size / step).toInt()
        if (outLen <= 1) return input
        val out = FloatArray(outLen)
        for (i in 0 until outLen) {
            val p = i * step
            val i0 = p.toInt()
            val i1 = min(i0 + 1, input.size - 1)
            val frac = p - i0
            out[i] = input[i0] * (1f - frac) + input[i1] * frac
        }
        return out
    }

    /** Removes silence at the head/tail so layers line up nicely. */
    fun trimSilence(input: FloatArray, threshold: Float = 0.004f): FloatArray {
        if (input.isEmpty()) return input
        var start = 0
        while (start < input.size && abs(input[start]) < threshold) start++
        var end = input.size - 1
        while (end > start && abs(input[end]) < threshold) end--
        if (start >= end) return input
        return input.copyOfRange(start, end + 1)
    }

    /** Short fade in/out to kill clicks. */
    fun fade(buf: FloatArray, inMs: Int = 3, outMs: Int = 12): FloatArray {
        val fi = min(inMs * WavIO.SAMPLE_RATE / 1000, buf.size / 2)
        val fo = min(outMs * WavIO.SAMPLE_RATE / 1000, buf.size / 2)
        for (i in 0 until fi) buf[i] *= i.toFloat() / fi
        for (i in 0 until fo) buf[buf.size - 1 - i] *= i.toFloat() / fo
        return buf
    }

    /** One-pole low-pass. amount 0..1 (0 = untouched). */
    fun lowpass(buf: FloatArray, amount: Float): FloatArray {
        if (amount <= 0.001f) return buf
        val a = amount.coerceIn(0f, 0.95f)
        var prev = 0f
        for (i in buf.indices) {
            prev = prev * a + buf[i] * (1 - a)
            buf[i] = prev
        }
        return buf
    }

    /** Simple decay/boost envelope shaping. */
    fun applyDecay(buf: FloatArray, curve: Float): FloatArray {
        if (abs(curve) < 0.01f) return buf
        val n = buf.size.toFloat()
        for (i in buf.indices) {
            val t = i / n
            val g = Math.pow((1.0 - t).toDouble(), curve.toDouble()).toFloat()
            buf[i] *= (1f - curve.coerceIn(0f, 1f)) + curve.coerceIn(0f, 1f) * g
        }
        return buf
    }

    fun addNoise(buf: FloatArray, amount: Float, rnd: Random): FloatArray {
        if (amount <= 0.0001f) return buf
        for (i in buf.indices) buf[i] += (rnd.nextFloat() * 2f - 1f) * amount * abs(buf[i])
        return buf
    }

    fun normalize(buf: FloatArray, peak: Float = 0.9f): FloatArray {
        var mx = 0f
        for (v in buf) mx = max(mx, abs(v))
        if (mx < 1e-6f) return buf
        val g = peak / mx
        for (i in buf.indices) buf[i] *= g
        return buf
    }

    /** Mixes layers, each with its own gain and start offset in samples. */
    fun mix(layers: List<FloatArray>, gains: List<Float>, offsets: List<Int>): FloatArray {
        var total = 0
        for (i in layers.indices) total = max(total, offsets[i] + layers[i].size)
        if (total <= 0) return FloatArray(0)
        val out = FloatArray(total)
        for (i in layers.indices) {
            val l = layers[i]
            val off = offsets[i]
            val g = gains[i]
            for (j in l.indices) out[off + j] += l[j] * g
        }
        return out
    }
}
