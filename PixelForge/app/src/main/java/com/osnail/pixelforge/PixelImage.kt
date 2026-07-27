package com.osnail.pixelforge

import android.graphics.Bitmap
import java.io.File

const val GRID_SIZE = 16

class PixelImage {
    val data = IntArray(GRID_SIZE * GRID_SIZE) { Palette.TRANSPARENT_IDX }

    fun get(x: Int, y: Int): Int = data[y * GRID_SIZE + x]
    fun set(x: Int, y: Int, v: Int) { data[y * GRID_SIZE + x] = v }
    fun clear() { for (i in data.indices) data[i] = Palette.TRANSPARENT_IDX }
    fun isBlank(): Boolean = data.all { it == Palette.TRANSPARENT_IDX }

    fun toBitmap(): Bitmap {
        val bmp = Bitmap.createBitmap(GRID_SIZE, GRID_SIZE, Bitmap.Config.ARGB_8888)
        for (y in 0 until GRID_SIZE) for (x in 0 until GRID_SIZE) {
            bmp.setPixel(x, y, Palette.colors[data[y * GRID_SIZE + x]])
        }
        return bmp
    }

    fun write(file: File) {
        val bytes = ByteArray(data.size) { data[it].toByte() }
        file.writeBytes(bytes)
    }

    companion object {
        fun fromIntArray(arr: IntArray): PixelImage {
            val img = PixelImage()
            for (i in arr.indices) if (i < img.data.size) img.data[i] = arr[i]
            return img
        }

        fun read(file: File): PixelImage {
            val img = PixelImage()
            val bytes = file.readBytes()
            for (i in img.data.indices) {
                if (i < bytes.size) img.data[i] = (bytes[i].toInt() and 0xFF).coerceIn(0, Palette.SIZE - 1)
            }
            return img
        }
    }
}
