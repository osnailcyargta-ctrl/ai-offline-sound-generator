package com.osnail.soundforge

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder

/**
 * Decodes any audio file the device can play (wav / mp3 / ogg / m4a / flac)
 * into mono 44.1kHz float PCM. Runs fully offline on the phone.
 */
object AudioDecoder {

    const val TARGET_SR = 44100
    private const val MAX_SECONDS = 20

    // MediaFormat.KEY_PCM_ENCODING / AudioFormat.ENCODING_* are API 24+ but the
    // constants are plain values; keeping local copies avoids the extra imports.
    private const val KEY_PCM_ENCODING = "pcm-encoding"
    private const val ENCODING_PCM_16BIT = 2
    private const val ENCODING_PCM_FLOAT = 4

    private fun readInt(format: MediaFormat, key: String, fallback: Int): Int =
        try {
            if (format.containsKey(key)) format.getInteger(key) else fallback
        } catch (e: Exception) {
            fallback
        }

    fun decode(ctx: Context, uri: Uri): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(ctx, uri, null)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) { trackIndex = i; format = f; break }
        }
        if (trackIndex < 0 || format == null) {
            extractor.release()
            throw IllegalArgumentException("No audio track in file")
        }
        extractor.selectTrack(trackIndex)

        var srcSr = readInt(format, MediaFormat.KEY_SAMPLE_RATE, TARGET_SR).coerceAtLeast(1)
        var channels = readInt(format, MediaFormat.KEY_CHANNEL_COUNT, 1).coerceAtLeast(1)
        var pcmFloat = readInt(format, KEY_PCM_ENCODING, ENCODING_PCM_16BIT) == ENCODING_PCM_FLOAT

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val buf = FloatBuf(srcSr * 2)
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var maxSamples = MAX_SECONDS * srcSr

        try {
            while (!outputDone && buf.size < maxSamples) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10000)
                    if (inIdx >= 0) {
                        val inBuf = codec.getInputBuffer(inIdx)!!
                        val sampleSize = extractor.readSampleData(inBuf, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(info, 10000)
                if (outIdx >= 0) {
                    if (info.size > 0) {
                        val outBuf = codec.getOutputBuffer(outIdx)!!
                        outBuf.position(info.offset)
                        outBuf.limit(info.offset + info.size)
                        outBuf.order(ByteOrder.LITTLE_ENDIAN)
                        if (pcmFloat) {
                            val fb = outBuf.asFloatBuffer()
                            val frames = fb.remaining() / channels
                            for (f in 0 until frames) {
                                var sum = 0f
                                for (c in 0 until channels) sum += fb.get()
                                buf.add(sum / channels)
                            }
                        } else {
                            val sb = outBuf.asShortBuffer()
                            val frames = sb.remaining() / channels
                            for (f in 0 until frames) {
                                var sum = 0f
                                for (c in 0 until channels) sum += sb.get() / 32768f
                                buf.add(sum / channels)
                            }
                        }
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    // The decoder is the authority on the real output layout; the
                    // extractor's track format is only a hint.
                    val nf = codec.outputFormat
                    srcSr = readInt(nf, MediaFormat.KEY_SAMPLE_RATE, srcSr).coerceAtLeast(1)
                    channels = readInt(nf, MediaFormat.KEY_CHANNEL_COUNT, channels).coerceAtLeast(1)
                    pcmFloat = readInt(nf, KEY_PCM_ENCODING, ENCODING_PCM_16BIT) == ENCODING_PCM_FLOAT
                    maxSamples = MAX_SECONDS * srcSr
                }
            }
        } finally {
            try { codec.stop() } catch (e: Exception) {}
            codec.release()
            extractor.release()
        }

        val mono = buf.toArray()
        return if (srcSr == TARGET_SR) mono else Dsp.resample(mono, srcSr.toFloat() / TARGET_SR)
    }
}
