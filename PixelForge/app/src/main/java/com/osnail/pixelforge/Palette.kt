package com.osnail.pixelforge

/**
 * A fixed palette keeps the model's output layer small (softmax over 24
 * classes instead of continuous RGB regression) and keeps pixel art crisp
 * instead of blurry. Index 0 is transparent/background and doubles as the
 * "nothing here yet" sentinel fed to the model at row/column edges.
 */
object Palette {
    val colors: IntArray = intArrayOf(
        0x00000000, // 0 transparent
        0xFF1A1A1A.toInt(), // 1 black
        0xFFFFFFFF.toInt(), // 2 white
        0xFF7F7F7F.toInt(), // 3 gray
        0xFFC0C0C0.toInt(), // 4 light gray
        0xFF3F3F3F.toInt(), // 5 dark gray
        0xFFE0A080.toInt(), // 6 skin light
        0xFFA8674A.toInt(), // 7 skin dark
        0xFFE74C3C.toInt(), // 8 red
        0xFF8B0000.toInt(), // 9 dark red
        0xFFFFA500.toInt(), // 10 orange
        0xFFF1C40F.toInt(), // 11 yellow
        0xFF2ECC71.toInt(), // 12 green
        0xFF145A32.toInt(), // 13 dark green
        0xFF00CED1.toInt(), // 14 cyan
        0xFF3498DB.toInt(), // 15 blue
        0xFF1B2A6B.toInt(), // 16 dark blue
        0xFF8E44AD.toInt(), // 17 purple
        0xFFFF00FF.toInt(), // 18 magenta
        0xFFFFC0CB.toInt(), // 19 pink
        0xFF8B4513.toInt(), // 20 brown
        0xFFD2B48C.toInt(), // 21 tan
        0xFF87CEEB.toInt(), // 22 sky blue
        0xFFFFD700.toInt()  // 23 gold
    )
    const val SIZE = 24
    const val TRANSPARENT_IDX = 0
}
