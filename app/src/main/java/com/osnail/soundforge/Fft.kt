package com.osnail.soundforge

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.PI

object Fft {
    /** In-place radix-2 Cooley-Tukey. n must be a power of 2. */
    fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j or bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wr = cos(ang); val wi = sin(ang)
            var i = 0
            while (i < n) {
                var curWr = 1.0; var curWi = 0.0
                val half = len / 2
                for (k in 0 until half) {
                    val uRe = re[i + k]; val uIm = im[i + k]
                    val vRe = re[i + k + half] * curWr - im[i + k + half] * curWi
                    val vIm = re[i + k + half] * curWi + im[i + k + half] * curWr
                    re[i + k] = uRe + vRe; im[i + k] = uIm + vIm
                    re[i + k + half] = uRe - vRe; im[i + k + half] = uIm - vIm
                    val nWr = curWr * wr - curWi * wi
                    val nWi = curWr * wi + curWi * wr
                    curWr = nWr; curWi = nWi
                }
                i += len
            }
            len = len shl 1
        }
    }

    fun ifft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        for (i in 0 until n) im[i] = -im[i]
        fft(re, im)
        for (i in 0 until n) { re[i] = re[i] / n; im[i] = -im[i] / n }
    }
}

