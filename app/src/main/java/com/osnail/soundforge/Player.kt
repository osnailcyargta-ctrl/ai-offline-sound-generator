package com.osnail.soundforge

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

object Player {

    /** Static-mode tracks live in shared memory, so cap what we hand them. */
    private const val MAX_SECONDS = 60

    private var track: AudioTrack? = null

    fun play(pcm: FloatArray) {
        stop()
        if (pcm.isEmpty()) return

        val count = minOf(pcm.size, MAX_SECONDS * WavIO.SAMPLE_RATE)
        val shorts = ShortArray(count)
        for (i in 0 until count) {
            var v = (pcm[i] * 32767f).toInt()
            if (v > 32767) v = 32767
            if (v < -32768) v = -32768
            shorts[i] = v.toShort()
        }

        // MODE_STATIC plays exactly bufferSizeInBytes worth of frames, so the
        // buffer has to be the sample count itself — padding it out to
        // getMinBufferSize() would append silence to every short sound.
        val bytes = (shorts.size * 2).coerceAtLeast(2)

        val t = try {
            AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(WavIO.SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(bytes)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()
        } catch (e: Exception) {
            // Out of shared memory / unsupported format — stay silent instead of
            // taking the whole activity down.
            return
        }

        if (t.state == AudioTrack.STATE_UNINITIALIZED) {
            t.release()
            return
        }

        try {
            t.write(shorts, 0, shorts.size)
            t.play()
            track = t
        } catch (e: Exception) {
            try { t.release() } catch (ignored: Exception) {}
        }
    }

    fun stop() {
        val t = track
        track = null
        try {
            t?.stop()
        } catch (e: Exception) {
        }
        try {
            t?.release()
        } catch (e: Exception) {
        }
    }
}
