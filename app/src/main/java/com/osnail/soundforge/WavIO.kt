package com.osnail.soundforge

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream

/** Growable float buffer without boxing. */
class FloatBuf(initial: Int = 8192) {
    var data = FloatArray(initial)
        private set
    var size = 0
        private set

    fun add(v: Float) {
        if (size == data.size) data = data.copyOf(data.size * 2)
        data[size++] = v
    }

    fun toArray(): FloatArray = data.copyOf(size)
}

object WavIO {

    const val SAMPLE_RATE = 44100

    fun write(file: File, pcm: FloatArray, sr: Int = SAMPLE_RATE) {
        file.outputStream().use { write(it, pcm, sr) }
    }

    fun toBytes(pcm: FloatArray, sr: Int = SAMPLE_RATE): ByteArray {
        val bos = ByteArrayOutputStream()
        write(bos, pcm, sr)
        return bos.toByteArray()
    }

    fun write(out: OutputStream, pcm: FloatArray, sr: Int = SAMPLE_RATE) {
        val dataLen = pcm.size * 2
        val header = ByteArray(44)
        fun putStr(off: Int, s: String) { for (i in s.indices) header[off + i] = s[i].code.toByte() }
        fun putInt(off: Int, v: Int) {
            header[off] = (v and 0xFF).toByte()
            header[off + 1] = ((v shr 8) and 0xFF).toByte()
            header[off + 2] = ((v shr 16) and 0xFF).toByte()
            header[off + 3] = ((v shr 24) and 0xFF).toByte()
        }
        fun putShort(off: Int, v: Int) {
            header[off] = (v and 0xFF).toByte()
            header[off + 1] = ((v shr 8) and 0xFF).toByte()
        }
        putStr(0, "RIFF"); putInt(4, 36 + dataLen); putStr(8, "WAVE")
        putStr(12, "fmt "); putInt(16, 16); putShort(20, 1); putShort(22, 1)
        putInt(24, sr); putInt(28, sr * 2); putShort(32, 2); putShort(34, 16)
        putStr(36, "data"); putInt(40, dataLen)
        out.write(header)

        val chunk = ByteArray(4096)
        var idx = 0
        var i = 0
        while (i < pcm.size) {
            var v = (pcm[i] * 32767f).toInt()
            if (v > 32767) v = 32767
            if (v < -32768) v = -32768
            chunk[idx++] = (v and 0xFF).toByte()
            chunk[idx++] = ((v shr 8) and 0xFF).toByte()
            if (idx == chunk.size) { out.write(chunk, 0, idx); idx = 0 }
            i++
        }
        if (idx > 0) out.write(chunk, 0, idx)
        out.flush()
    }

    /** Reads a 16-bit PCM mono/stereo wav. Returns mono float samples. */
    fun read(file: File): FloatArray = file.inputStream().use { read(it) }

    fun read(input: InputStream): FloatArray {
        val bytes = input.readBytes()
        if (bytes.size < 44) return FloatArray(0)
        fun i32(o: Int) = (bytes[o].toInt() and 0xFF) or ((bytes[o + 1].toInt() and 0xFF) shl 8) or
                ((bytes[o + 2].toInt() and 0xFF) shl 16) or ((bytes[o + 3].toInt() and 0xFF) shl 24)
        fun i16(o: Int) = ((bytes[o].toInt() and 0xFF) or ((bytes[o + 1].toInt() and 0xFF) shl 8)).toShort().toInt()

        var pos = 12
        var channels = 1
        var bits = 16
        var dataOff = -1
        var dataLen = 0
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val len = i32(pos + 4)
            when (id) {
                "fmt " -> { channels = i16(pos + 10); bits = i16(pos + 22) }
                "data" -> { dataOff = pos + 8; dataLen = minOf(len, bytes.size - dataOff) }
            }
            if (dataOff >= 0) break
            pos += 8 + len + (len and 1)
        }
        if (dataOff < 0 || bits != 16) return FloatArray(0)

        val frames = dataLen / 2 / channels
        val out = FloatArray(frames)
        var o = dataOff
        for (f in 0 until frames) {
            var sum = 0f
            for (c in 0 until channels) {
                sum += i16(o) / 32768f
                o += 2
            }
            out[f] = sum / channels
        }
        return out
    }
}
