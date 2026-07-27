package com.osnail.pixelforge

import kotlin.random.Random

data class ImgGenResult(val image: PixelImage?, val log: String)

/**
 * A single matched category generates straight from the model. Multiple
 * matched categories blend their learned embeddings into ONE generation pass
 * -- a genuine model-level concept blend, not a pixel-by-pixel cross-fade of
 * two finished images (which would just look like ghosting).
 */
object Generator {

    fun generate(model: PixelGru, matches: List<Match>, variation: Float, seed: Long = System.nanoTime()): ImgGenResult {
        val valid = matches.filter { model.categories.contains(it.category) }
        if (valid.isEmpty()) {
            val skipped = matches.joinToString(", ") { it.category }
            return ImgGenResult(null, "No usable categories (not in trained model): $skipped")
        }

        val temperature = (0.15 + 0.9 * variation.coerceIn(0f, 1f)).coerceIn(0.1, 1.2)
        val rnd = Random(seed)

        val pixels: IntArray
        val log: String
        if (valid.size == 1) {
            val m = valid[0]
            val idx = model.categoryIndex(m.category)
            pixels = model.generate(idx, temperature = temperature, seed = rnd.nextLong())
            log = "${m.category} (match ${(m.score * 100).toInt()}%, AI-generated)"
        } else {
            val idxs = valid.map { model.categoryIndex(it.category) }
            val weights = valid.map { it.score }
            pixels = model.generateBlended(idxs, weights, temperature = temperature, seed = rnd.nextLong())
            val names = valid.joinToString(" + ") { "${it.category} (${(it.score * 100).toInt()}%)" }
            log = "Blended: $names — AI-synthesized from a combined concept, not a pixel overlay"
        }

        return ImgGenResult(PixelImage.fromIntArray(pixels), log)
    }
}
