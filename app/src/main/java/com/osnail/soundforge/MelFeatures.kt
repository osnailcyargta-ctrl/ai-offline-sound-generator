package com.osnail.soundforge

import kotlin.math.*

object MelFeatures {
    const val N_FFT = 2048
    const val HOP = 512
    const val N_MELS = 80
    const val FMIN = 20.0

    private fun hann(n: Int): DoubleArray = DoubleArray(n) { 0.5 - 0.5 * cos(2 * PI * it / n) }

    // ---------------------------------------------------------- STFT/ISTFT

    fun stft(x: FloatArray, nFft: Int = N_FFT, hop: Int = HOP): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        val win = hann(nFft)
        val pad = nFft / 2
        val padded = DoubleArray(x.size + 2 * pad)
        for (i in x.indices) padded[i + pad] = x[i].toDouble()
        // reflect padding at the edges (cheap edge mirror, good enough here)
        for (i in 0 until pad) {
            padded[pad - 1 - i] = if (i < x.size) x[i].toDouble() else 0.0
            val srcIdx = x.size - 1 - i
            padded[pad + x.size + i] = if (srcIdx >= 0) x[srcIdx].toDouble() else 0.0
        }

        var nFrames = 1 + (padded.size - nFft) / hop
        if (nFrames < 1) nFrames = 1
        val freqBins = nFft / 2 + 1
        val re = Array(freqBins) { DoubleArray(nFrames) }
        val im = Array(freqBins) { DoubleArray(nFrames) }

        val fre = DoubleArray(nFft)
        val fim = DoubleArray(nFft)
        for (f in 0 until nFrames) {
            val start = f * hop
            for (i in 0 until nFft) {
                val idx = start + i
                fre[i] = (if (idx < padded.size) padded[idx] else 0.0) * win[i]
                fim[i] = 0.0
            }
            Fft.fft(fre, fim)
            for (k in 0 until freqBins) {
                re[k][f] = fre[k]
                im[k][f] = fim[k]
            }
        }
        return Pair(re, im)
    }

    fun istft(re: Array<DoubleArray>, im: Array<DoubleArray>, hop: Int = HOP): FloatArray {
        val freqBins = re.size
        val nFft = (freqBins - 1) * 2
        val nFrames = re[0].size
        val win = hann(nFft)
        val outLen = nFft + hop * (nFrames - 1)
        val out = DoubleArray(outLen)
        val norm = DoubleArray(outLen)

        val fre = DoubleArray(nFft)
        val fim = DoubleArray(nFft)
        for (f in 0 until nFrames) {
            // rebuild full spectrum via conjugate symmetry
            for (k in 0 until freqBins) { fre[k] = re[k][f]; fim[k] = im[k][f] }
            for (k in 1 until nFft - freqBins + 1) {
                fre[freqBins - 1 + k] = re[freqBins - 1 - k][f]
                fim[freqBins - 1 + k] = -im[freqBins - 1 - k][f]
            }
            Fft.ifft(fre, fim)
            val start = f * hop
            for (i in 0 until nFft) {
                out[start + i] += fre[i] * win[i]
                norm[start + i] += win[i] * win[i]
            }
        }
        for (i in out.indices) if (norm[i] > 1e-8) out[i] /= norm[i]

        val pad = nFft / 2
        val trimmed = if (outLen > 2 * pad) out.copyOfRange(pad, outLen - pad) else out
        return FloatArray(trimmed.size) { trimmed[it].toFloat() }
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
            for (k in left until center) fb[m - 1][k] = (k - left).toDouble() / max(1, center - left)
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
        val (re, im) = stft(x, nFft, hop)
        val freqBins = re.size; val nFrames = re[0].size
        val mag = Array(freqBins) { DoubleArray(nFrames) }
        for (k in 0 until freqBins) for (f in 0 until nFrames) mag[k][f] = hypot(re[k][f], im[k][f])

        val mel = matMul(melBasis, mag) // (nMels x nFrames)
        val out = Array(nFrames) { FloatArray(nMels) }
        for (f in 0 until nFrames) for (m in 0 until nMels) {
            out[f][m] = ln(max(mel[m][f], 1e-5)).toFloat()
        }
        return out
    }

    fun melToAudio(logMel: Array<FloatArray>, sr: Int, nFft: Int = N_FFT, hop: Int = HOP, nIter: Int = 24): FloatArray {
        ensureBasis(sr)
        val nFrames = logMel.size
        val nMels = if (nFrames > 0) logMel[0].size else N_MELS
        val mel = Array(nMels) { m -> DoubleArray(nFrames) { f -> exp(logMel[f][m].toDouble()) } }
        val magFull = matMul(melPinv, mel) // (nFreqs x nFrames)
        val freqBins = magFull.size
        for (k in 0 until freqBins) for (f in 0 until nFrames) magFull[k][f] = max(magFull[k][f], 0.0)

        val rnd = java.util.Random()
        var re = Array(freqBins) { k -> DoubleArray(nFrames) { f -> magFull[k][f] * cos(rnd.nextDouble() * 2 * PI) } }
        var im = Array(freqBins) { k -> DoubleArray(nFrames) { f -> 0.0 } }
        for (k in 0 until freqBins) for (f in 0 until nFrames) {
            val phase = rnd.nextDouble() * 2 * PI
            re[k][f] = magFull[k][f] * cos(phase)
            im[k][f] = magFull[k][f] * sin(phase)
        }

        var audio = FloatArray(0)
        repeat(nIter) {
            audio = istft(re, im, hop)
            val (rre, rim) = stft(audio, nFft, hop)
            val n = min(rre[0].size, nFrames)
            val newRe = Array(freqBins) { DoubleArray(nFrames) }
            val newIm = Array(freqBins) { DoubleArray(nFrames) }
            for (k in 0 until freqBins) for (f in 0 until n) {
                val mag = hypot(rre[k][f], rim[k][f]).let { if (it < 1e-8) 1e-8 else it }
                val ratio = magFull[k][f] / mag
                newRe[k][f] = rre[k][f] * ratio
                newIm[k][f] = rim[k][f] * ratio
            }
            re = newRe; im = newIm
        }
        return istft(re, im, hop)
    }
}

