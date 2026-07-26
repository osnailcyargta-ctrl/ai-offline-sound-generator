package com.osnail.soundforge

import kotlin.math.*

object MelFeatures {
    const val N_FFT = 2048
    const val HOP = 512
    const val N_MELS = 80
    const val FMIN = 20.0

    private fun hann(n: Int): DoubleArray = DoubleArray(n) { 0.5 - 0.5 * cos(2 * PI * it / n) }

    /** Number of STFT frames a signal of [len] samples produces. */
    private fun frameCount(len: Int, nFft: Int, hop: Int): Int =
        maxOf(1, 1 + (len + 2 * (nFft / 2) - nFft) / hop)

    // ---------------------------------------------------------- STFT workspace

    /**
     * Every buffer an STFT/ISTFT of one fixed (nFft, hop, nFrames) shape needs,
     * allocated once and overwritten in place.
     *
     * Spectrograms are flat DoubleArrays indexed `[k * nFrames + f]` rather than
     * `Array<DoubleArray>`, so a spectrogram is one object instead of freqBins of
     * them. Combined with reusing these buffers, a 24-iteration Griffin-Lim run
     * costs the same memory as a single iteration — it used to allocate a fresh
     * set on every pass.
     */
    private class Stft(val nFft: Int, val hop: Int, val nFrames: Int, signalLen: Int) {
        val pad = nFft / 2
        val freqBins = nFft / 2 + 1
        val specSize = freqBins * nFrames
        val outLen = nFft + hop * (nFrames - 1)
        val audioLen = if (outLen > 2 * pad) outLen - 2 * pad else outLen

        private val win = hann(nFft)
        private val fre = DoubleArray(nFft)
        private val fim = DoubleArray(nFft)
        private val padded = DoubleArray(maxOf(signalLen, audioLen) + 2 * pad)
        private val acc = DoubleArray(outLen)
        private val norm = DoubleArray(outLen)

        /** Shared output of [synthesize]; valid until the next call. */
        val audio = FloatArray(audioLen)

        fun spectrogram() = DoubleArray(specSize)

        /** STFT of the first [len] samples of [x], written into [re] and [im]. */
        fun analyze(x: FloatArray, len: Int, re: DoubleArray, im: DoubleArray) {
            java.util.Arrays.fill(padded, 0.0)
            for (i in 0 until len) padded[i + pad] = x[i].toDouble()
            // reflect padding at the edges (cheap edge mirror, good enough here)
            for (i in 0 until pad) {
                padded[pad - 1 - i] = if (i < len) x[i].toDouble() else 0.0
                val srcIdx = len - 1 - i
                val dstIdx = pad + len + i
                if (dstIdx < padded.size) {
                    padded[dstIdx] = if (srcIdx >= 0) x[srcIdx].toDouble() else 0.0
                }
            }

            for (f in 0 until nFrames) {
                val start = f * hop
                for (i in 0 until nFft) {
                    val idx = start + i
                    fre[i] = (if (idx < padded.size) padded[idx] else 0.0) * win[i]
                    fim[i] = 0.0
                }
                Fft.fft(fre, fim)
                for (k in 0 until freqBins) {
                    re[k * nFrames + f] = fre[k]
                    im[k * nFrames + f] = fim[k]
                }
            }
        }

        /** ISTFT of [re]/[im] into [audio], which it returns. */
        fun synthesize(re: DoubleArray, im: DoubleArray): FloatArray {
            java.util.Arrays.fill(acc, 0.0)
            java.util.Arrays.fill(norm, 0.0)
            for (f in 0 until nFrames) {
                // rebuild full spectrum via conjugate symmetry
                for (k in 0 until freqBins) {
                    fre[k] = re[k * nFrames + f]
                    fim[k] = im[k * nFrames + f]
                }
                for (k in 1 until nFft - freqBins + 1) {
                    fre[freqBins - 1 + k] = re[(freqBins - 1 - k) * nFrames + f]
                    fim[freqBins - 1 + k] = -im[(freqBins - 1 - k) * nFrames + f]
                }
                Fft.ifft(fre, fim)
                val start = f * hop
                for (i in 0 until nFft) {
                    acc[start + i] += fre[i] * win[i]
                    norm[start + i] += win[i] * win[i]
                }
            }
            for (i in acc.indices) if (norm[i] > 1e-8) acc[i] /= norm[i]

            val off = if (outLen > 2 * pad) pad else 0
            for (i in 0 until audioLen) audio[i] = acc[off + i].toFloat()
            return audio
        }
    }

    // ---------------------------------------------------------- mel filterbank

    private fun hzToMel(f: Double) = 2595.0 * log10(1.0 + f / 700.0)
    private fun melToHz(m: Double) = 700.0 * (10.0.pow(m / 2595.0) - 1.0)

    fun melFilterbank(sr: Int, nFft: Int = N_FFT, nMels: Int = N_MELS, fmin: Double = FMIN, fmax: Double = sr / 2.0): Array<DoubleArray> {
        val nFreqs = nFft / 2 + 1
        val melMin = hzToMel(fmin); val melMax = hzToMel(fmax)
        val melPts = DoubleArray(nMels + 2) { melMin + (melMax - melMin) * it / (nMels + 1) }
        val hzPts = melPts.map { melToHz(it) }
        val binPts = hzPts.map { (floor((nFft + 1) * it / sr)).toInt().coerceIn(0, nFreqs - 1) }

        val fb = Array(nMels) { DoubleArray(nFreqs) }
        for (m in 1..nMels) {
            var left = binPts[m - 1]; var center = binPts[m]; var right = binPts[m + 1]
            if (center == left) center += 1
            if (right == center) right += 1
            for (k in left until center) if (k < nFreqs) fb[m - 1][k] = (k - left).toDouble() / max(1, center - left)
            for (k in center until right) if (k < nFreqs) fb[m - 1][k] = (right - k).toDouble() / max(1, right - center)
        }
        return fb
    }

    // ---------------------------------------------------------- small matrix ops

    fun matMul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size; val k = a[0].size; val m = b[0].size
        val out = Array(n) { DoubleArray(m) }
        for (i in 0 until n) for (p in 0 until k) {
            val aip = a[i][p]
            if (aip == 0.0) continue
            for (j in 0 until m) out[i][j] += aip * b[p][j]
        }
        return out
    }

    fun transpose(a: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size; val m = a[0].size
        val out = Array(m) { DoubleArray(n) }
        for (i in 0 until n) for (j in 0 until m) out[j][i] = a[i][j]
        return out
    }

    /** Gauss-Jordan inverse of a small square matrix. */
    fun invert(a: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size
        val m = Array(n) { i -> DoubleArray(2 * n).also { row ->
            for (j in 0 until n) row[j] = a[i][j]
            row[n + i] = 1.0
        } }
        for (col in 0 until n) {
            var pivot = col
            for (r in col until n) if (abs(m[r][col]) > abs(m[pivot][col])) pivot = r
            val tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp
            val pv = m[col][col].let { if (abs(it) < 1e-12) 1e-12 else it }
            for (j in 0 until 2 * n) m[col][j] /= pv
            for (r in 0 until n) {
                if (r == col) continue
                val factor = m[r][col]
                if (factor == 0.0) continue
                for (j in 0 until 2 * n) m[r][j] -= factor * m[col][j]
            }
        }
        return Array(n) { i -> DoubleArray(n) { j -> m[i][n + j] } }
    }

    // pseudo-inverse of A (nMels x nFreqs, full row rank): pinv(A) = A^T (A A^T)^-1
    fun pseudoInverse(a: Array<DoubleArray>): Array<DoubleArray> {
        val at = transpose(a)
        val aat = matMul(a, at)
        val aatInv = invert(aat)
        return matMul(at, aatInv) // (nFreqs x nMels)
    }

    // ---------------------------------------------------------- public API

    private var cachedSr = -1
    private lateinit var melBasis: Array<DoubleArray>
    private lateinit var melPinv: Array<DoubleArray>

    private fun ensureBasis(sr: Int) {
        if (cachedSr == sr) return
        melBasis = melFilterbank(sr)
        melPinv = pseudoInverse(melBasis)
        cachedSr = sr
    }

    fun audioToMel(x: FloatArray, sr: Int, nFft: Int = N_FFT, hop: Int = HOP, nMels: Int = N_MELS): Array<FloatArray> {
        ensureBasis(sr)
        val nFrames = frameCount(x.size, nFft, hop)
        val st = Stft(nFft, hop, nFrames, x.size)

        val re = st.spectrogram()
        val im = st.spectrogram()
        st.analyze(x, x.size, re, im)
        // Magnitude overwrites `re` instead of filling a third spectrogram.
        for (i in re.indices) re[i] = hypot(re[i], im[i])

        val mel = DoubleArray(nMels * nFrames)
        for (m in 0 until nMels) {
            val row = melBasis[m]
            val mBase = m * nFrames
            for (k in 0 until st.freqBins) {
                val w = row[k]
                if (w == 0.0) continue
                val kBase = k * nFrames
                for (f in 0 until nFrames) mel[mBase + f] += w * re[kBase + f]
            }
        }
        return Array(nFrames) { f -> FloatArray(nMels) { m -> ln(max(mel[m * nFrames + f], 1e-5)).toFloat() } }
    }

    fun melToAudio(logMel: Array<FloatArray>, sr: Int, nFft: Int = N_FFT, hop: Int = HOP, nIter: Int = 24): FloatArray {
        ensureBasis(sr)
        val nFrames = logMel.size
        if (nFrames == 0) return FloatArray(0)
        val nMels = logMel[0].size
        val st = Stft(nFft, hop, nFrames, hop * maxOf(0, nFrames - 1))
        val freqBins = st.freqBins

        val mel = DoubleArray(nMels * nFrames)
        for (m in 0 until nMels) for (f in 0 until nFrames) {
            mel[m * nFrames + f] = exp(logMel[f][m].toDouble())
        }

        val magFull = DoubleArray(st.specSize)
        for (k in 0 until freqBins) {
            val row = melPinv[k]
            val kBase = k * nFrames
            for (m in 0 until nMels) {
                val w = row[m]
                if (w == 0.0) continue
                val mBase = m * nFrames
                for (f in 0 until nFrames) magFull[kBase + f] += w * mel[mBase + f]
            }
        }
        for (i in magFull.indices) if (magFull[i] < 0.0) magFull[i] = 0.0

        val re = st.spectrogram()
        val im = st.spectrogram()
        val rnd = java.util.Random()
        for (i in 0 until st.specSize) {
            val phase = rnd.nextDouble() * 2 * PI
            re[i] = magFull[i] * cos(phase)
            im[i] = magFull[i] * sin(phase)
        }

        // Griffin-Lim: resynthesize, re-analyze, snap magnitudes back to the
        // target. `re`/`im` are rewritten in place on every pass.
        repeat(nIter) {
            val audio = st.synthesize(re, im)
            st.analyze(audio, audio.size, re, im)
            for (i in 0 until st.specSize) {
                var mag = hypot(re[i], im[i])
                if (mag < 1e-8) mag = 1e-8
                val ratio = magFull[i] / mag
                re[i] *= ratio
                im[i] *= ratio
            }
        }
        // synthesize() hands back its shared buffer, so the caller gets a copy.
        return st.synthesize(re, im).copyOf()
    }
}
