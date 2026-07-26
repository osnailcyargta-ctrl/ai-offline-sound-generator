package com.osnail.soundforge

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.ln
import kotlin.math.sqrt

data class Match(val category: String, val score: Double, val token: String)

object TextUtil {
    private val STOP = hashSetOf(
        "buatin", "buat", "bikin", "buatkan", "bikinin", "tolong", "dong", "coba", "kasih",
        "suara", "sound", "sounds", "effect", "effects", "efek", "sfx", "audio", "bunyi",
        "generate", "make", "create", "please", "give", "want", "need",
        "gw", "gue", "aku", "saya", "kita", "yang", "itu", "ini", "dan", "plus", "campur",
        "mix", "with", "dengan", "and", "the", "for", "from", "into", "some", "like",
        "kayak", "seperti", "sebuah", "sama", "biar", "jadi", "pake", "pakai", "buatlah"
    )

    fun tokenize(text: String): List<String> =
        text.lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length >= 2 && it !in STOP }

    fun ngrams(word: String, n: Int): List<String> {
        val pad = " ".repeat(n - 1)
        val s = pad + word + pad
        if (s.length < n) return listOf(s)
        val out = ArrayList<String>(s.length - n + 1)
        for (i in 0..s.length - n) out.add(s.substring(i, i + n))
        return out
    }
}

/**
 * Trains on the category names + aliases the user typed.
 * Matching is character n-gram cosine similarity with idf weighting,
 * so "bom" still finds "boom" and "lasser" still finds "laser".
 */
class NgramModel(val n: Int = 3) {

    private val vectors = HashMap<String, HashMap<String, Double>>()
    private val idf = HashMap<String, Double>()
    private val exactWords = HashMap<String, HashSet<String>>()
    var vocabSize = 0; private set
    var trainedCategories = 0; private set

    fun train(categories: List<Category>) {
        vectors.clear(); idf.clear(); exactWords.clear()
        val df = HashMap<String, Int>()

        for (cat in categories) {
            val words = ArrayList<String>()
            words.addAll(TextUtil.tokenize(cat.name))
            for (a in cat.aliases) words.addAll(TextUtil.tokenize(a))
            if (words.isEmpty()) words.add(cat.name.lowercase())

            val vec = HashMap<String, Double>()
            for (w in words) {
                exactWords.getOrPut(w) { HashSet() }.add(cat.name)
                for (g in TextUtil.ngrams(w, n)) vec[g] = (vec[g] ?: 0.0) + 1.0
            }
            vectors[cat.name] = vec
            for (g in vec.keys) df[g] = (df[g] ?: 0) + 1
        }

        val total = categories.size.coerceAtLeast(1)
        for ((g, d) in df) idf[g] = ln(total.toDouble() / d) + 1.0
        vocabSize = df.size
        trainedCategories = categories.size
    }

    val isTrained: Boolean get() = vectors.isNotEmpty()

    private fun similarity(token: String, category: String): Double {
        val vec = vectors[category] ?: return 0.0
        val q = HashMap<String, Double>()
        for (g in TextUtil.ngrams(token, n)) q[g] = (q[g] ?: 0.0) + 1.0

        var dot = 0.0; var nq = 0.0; var nd = 0.0
        for ((g, v) in q) {
            val w = idf[g] ?: 1.0
            val a = v * w
            nq += a * a
            val b = (vec[g] ?: 0.0) * w
            if (b > 0) dot += a * b
        }
        for ((g, v) in vec) {
            val w = idf[g] ?: 1.0
            val b = v * w
            nd += b * b
        }
        if (nq <= 0 || nd <= 0) return 0.0
        return dot / (sqrt(nq) * sqrt(nd))
    }

    /** Returns the categories detected inside a free-form prompt. */
    fun match(prompt: String, threshold: Double = 0.34, maxLayers: Int = 4): List<Match> {
        val tokens = TextUtil.tokenize(prompt)
        if (tokens.isEmpty() || vectors.isEmpty()) return emptyList()

        val candidates = ArrayList<String>(tokens)
        for (i in 0 until tokens.size - 1) candidates.add(tokens[i] + tokens[i + 1])

        val best = HashMap<String, Match>()
        for (tok in candidates) {
            exactWords[tok]?.forEach { cat ->
                val cur = best[cat]
                if (cur == null || cur.score < 1.0) best[cat] = Match(cat, 1.0, tok)
            }
            for (cat in vectors.keys) {
                var s = similarity(tok, cat)
                val lc = cat.lowercase()
                if (lc.contains(tok) || tok.contains(lc)) s = maxOf(s, 0.82)
                if (s >= threshold) {
                    val cur = best[cat]
                    if (cur == null || s > cur.score) best[cat] = Match(cat, s, tok)
                }
            }
        }
        return best.values.sortedByDescending { it.score }.take(maxLayers)
    }

    fun toJson(): JSONObject {
        val root = JSONObject()
        root.put("n", n)
        root.put("vocab", vocabSize)
        val vecs = JSONObject()
        for ((cat, vec) in vectors) {
            val o = JSONObject()
            for ((g, v) in vec) o.put(g, v)
            vecs.put(cat, o)
        }
        root.put("vectors", vecs)
        val idfObj = JSONObject()
        for ((g, v) in idf) idfObj.put(g, v)
        root.put("idf", idfObj)
        val ex = JSONObject()
        for ((w, cats) in exactWords) ex.put(w, JSONArray(cats.toList()))
        root.put("words", ex)
        return root
    }

    companion object {
        fun fromJson(root: JSONObject): NgramModel {
            val m = NgramModel(root.optInt("n", 3))
            val vecs = root.optJSONObject("vectors") ?: JSONObject()
            for (cat in vecs.keys()) {
                val o = vecs.getJSONObject(cat)
                val map = HashMap<String, Double>()
                for (g in o.keys()) map[g] = o.getDouble(g)
                m.vectors[cat] = map
            }
            val idfObj = root.optJSONObject("idf") ?: JSONObject()
            for (g in idfObj.keys()) m.idf[g] = idfObj.getDouble(g)
            val ex = root.optJSONObject("words") ?: JSONObject()
            for (w in ex.keys()) {
                val arr = ex.getJSONArray(w)
                val set = HashSet<String>()
                for (i in 0 until arr.length()) set.add(arr.getString(i))
                m.exactWords[w] = set
            }
            m.vocabSize = root.optInt("vocab", m.idf.size)
            m.trainedCategories = m.vectors.size
            return m
        }
    }
}
