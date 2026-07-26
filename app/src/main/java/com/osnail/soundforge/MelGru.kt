package com.osnail.soundforge

import kotlin.math.*
import kotlin.random.Random

class MelGru(
    val nMels: Int = 80,
    val hidden: Int = 128,
    val embedDim: Int = 16,
    var categories: MutableList<String> = mutableListOf(),
    seed: Long = 0
) {
    private val rnd = Random(seed)
    val inDim = nMels + embedDim
    val H = hidden

    private fun init(rows: Int, cols: Int): Array<DoubleArray> {
        val scale = 1.0 / sqrt(cols.toDouble())
        return Array(rows) { DoubleArray(cols) { (rnd.nextDouble() * 2 - 1) * scale } }
    }
    private fun zeros(n: Int) = DoubleArray(n)
    private fun zerosM(r: Int, c: Int) = Array(r) { DoubleArray(c) }

    var Wz = init(H, inDim); var Uz = init(H, H); var bz = zeros(H)
    var Wr = init(H, inDim); var Ur = init(H, H); var br = zeros(H)
    var Wh = init(H, inDim); var Uh = init(H, H); var bh = zeros(H)
    var Wy = init(nMels, H); var by = zeros(nMels)
    var Emb = Array(max(1, categories.size)) { DoubleArray(embedDim) { (rnd.nextDouble() * 2 - 1) * 0.1 } }

    private var adamM: MutableMap<String, Any>? = null
    private var adamV: MutableMap<String, Any>? = null
    private var adamT = 0

    fun categoryIndex(name: String): Int {
        val i = categories.indexOf(name)
        return if (i < 0) 0 else i
    }

    // ---------------- vector/matrix helpers (small, hand-rolled) ----------

    private fun matVec(m: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        val out = DoubleArray(m.size)
        for (i in m.indices) {
            var s = 0.0
            val row = m[i]
            for (j in v.indices) s += row[j] * v[j]
            out[i] = s
        }
        return out
    }
    private fun add(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] + b[it] }
    private fun add3(a: DoubleArray, b: DoubleArray, c: DoubleArray) = DoubleArray(a.size) { a[it] + b[it] + c[it] }
    private fun mulElem(a: DoubleArray, b: DoubleArray) = DoubleArray(a.size) { a[it] * b[it] }
    private fun sigmoid(x: DoubleArray) = DoubleArray(x.size) { 1.0 / (1.0 + exp(-x[it].coerceIn(-30.0, 30.0))) }
    private fun tanhV(x: DoubleArray) = DoubleArray(x.size) { tanh(x[it]) }
    private fun concat(a: DoubleArray, b: DoubleArray) = DoubleArray(a.size + b.size) { if (it < a.size) a[it] else b[it - a.size] }
    private fun outer(a: DoubleArray, b: DoubleArray): Array<DoubleArray> = Array(a.size) { i -> DoubleArray(b.size) { j -> a[i] * b[j] } }
    private fun matVecT(m: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        // m^T @ v, m is (rows x cols) -> result length cols
        val cols = m[0].size
        val out = DoubleArray(cols)
        for (i in m.indices) {
            val vi = v[i]
            if (vi == 0.0) continue
            val row = m[i]
            for (j in 0 until cols) out[j] += row[j] * vi
        }
        return out
    }
    private fun addInPlace(m: Array<DoubleArray>, delta: Array<DoubleArray>) {
        for (i in m.indices) for (j in m[i].indices) m[i][j] += delta[i][j]
    }
    private fun addInPlaceV(v: DoubleArray, delta: DoubleArray) { for (i in v.indices) v[i] += delta[i] }

    data class GateCache(val z: DoubleArray, val r: DoubleArray, val hTilde: DoubleArray)

    private fun step(xt: DoubleArray, hPrev: DoubleArray): Triple<DoubleArray, DoubleArray, GateCache> {
        val z = sigmoid(add3(matVec(Wz, xt), matVec(Uz, hPrev), bz))
        val r = sigmoid(add3(matVec(Wr, xt), matVec(Ur, hPrev), br))
        val rh = mulElem(r, hPrev)
        val hTilde = tanhV(add3(matVec(Wh, xt), matVec(Uh, rh), bh))
        val h = DoubleArray(H) { (1 - z[it]) * hPrev[it] + z[it] * hTilde[it] }
        val y = add(matVec(Wy, h), by)
        return Triple(h, y, GateCache(z, r, hTilde))
    }

    fun trainSequence(melSeq: Array<FloatArray>, catIdx: Int, lr: Double = 0.003): Double {
        val T = melSeq.size
        if (T < 2) return 0.0
        val emb = Emb[catIdx]
        val n = T - 1

        val hs = ArrayList<DoubleArray>(n + 1)
        val ys = ArrayList<DoubleArray>(n)
        val caches = ArrayList<GateCache>(n)
        val xs = ArrayList<DoubleArray>(n)

        var h = DoubleArray(H)
        hs.add(h)
        var prevFrame = DoubleArray(nMels)
        for (t in 0 until n) {
            val xt = concat(prevFrame, emb)
            val (hNew, y, cache) = step(xt, h)
            h = hNew
            hs.add(h); ys.add(y); caches.add(cache); xs.add(xt)
            prevFrame = DoubleArray(nMels) { melSeq[t][it].toDouble() }
        }

        var loss = 0.0
        val diffs = ArrayList<DoubleArray>(n)
        for (t in 0 until n) {
            val target = DoubleArray(nMels) { melSeq[t + 1][it].toDouble() }
            val diff = DoubleArray(nMels) { ys[t][it] - target[it] }
            diffs.add(diff)
            for (v in diff) loss += v * v
        }
        loss /= (n * nMels)

        val gWz = zerosM(H, inDim); val gUz = zerosM(H, H); val gbz = zeros(H)
        val gWr = zerosM(H, inDim); val gUr = zerosM(H, H); val gbr = zeros(H)
        val gWh = zerosM(H, inDim); val gUh = zerosM(H, H); val gbh = zeros(H)
        val gWy = zerosM(nMels, H); val gby = zeros(nMels)
        val gEmb = zeros(embedDim)

        var dHNext = DoubleArray(H)
        for (t in n - 1 downTo 0) {
            val cache = caches[t]
            val z = cache.z; val r = cache.r; val hTilde = cache.hTilde
            val hPrev = hs[t]; val hT = hs[t + 1]; val xt = xs[t]

            val dy = DoubleArray(nMels) { (2.0 / (n * nMels)) * diffs[t][it] }
            addInPlace(gWy, outer(dy, hT))
            addInPlaceV(gby, dy)
            // Wy^T @ dy  (Wy is nMels x H, so Wy^T @ dy has length H)
            val dH2 = DoubleArray(H)
            for (i in 0 until nMels) {
                val d = dy[i]
                if (d == 0.0) continue
                val row = Wy[i]
                for (j in 0 until H) dH2[j] += row[j] * d
            }
            for (j in 0 until H) dH2[j] += dHNext[j]

            val dZ = DoubleArray(H) { dH2[it] * (hTilde[it] - hPrev[it]) }
            val dHTilde = DoubleArray(H) { dH2[it] * z[it] }
            val dHPrevDirect = DoubleArray(H) { dH2[it] * (1 - z[it]) }

            val dHTildeRaw = DoubleArray(H) { dHTilde[it] * (1 - hTilde[it] * hTilde[it]) }
            addInPlace(gWh, outer(dHTildeRaw, xt))
            val rh = mulElem(r, hPrev)
            addInPlace(gUh, outer(dHTildeRaw, rh))
            addInPlaceV(gbh, dHTildeRaw)

            // d(r*h_prev) = Uh^T @ dHTildeRaw
            val dRH = matVecT(Uh, dHTildeRaw)
            val dR = DoubleArray(H) { dRH[it] * hPrev[it] }
            val dRHprevPart = DoubleArray(H) { dRH[it] * r[it] }

            val dZRaw = DoubleArray(H) { dZ[it] * z[it] * (1 - z[it]) }
            addInPlace(gWz, outer(dZRaw, xt))
            addInPlace(gUz, outer(dZRaw, hPrev))
            addInPlaceV(gbz, dZRaw)

            val dRRaw = DoubleArray(H) { dR[it] * r[it] * (1 - r[it]) }
            addInPlace(gWr, outer(dRRaw, xt))
            addInPlace(gUr, outer(dRRaw, hPrev))
            addInPlaceV(gbr, dRRaw)

            val dXfromZ = matVecT(Wz, dZRaw)
            val dXfromR = matVecT(Wr, dRRaw)
            val dXfromH = matVecT(Wh, dHTildeRaw)
            for (k in 0 until embedDim) {
                gEmb[k] += dXfromZ[nMels + k] + dXfromR[nMels + k] + dXfromH[nMels + k]
            }

            val dUzT = matVecT(Uz, dZRaw)
            val dUrT = matVecT(Ur, dRRaw)
            dHNext = DoubleArray(H) { dHPrevDirect[it] + dUzT[it] + dUrT[it] + dRHprevPart[it] }
        }

        adamStep(
            mapOf(
                "Wz" to gWz, "Uz" to gUz, "Wr" to gWr, "Ur" to gUr,
                "Wh" to gWh, "Uh" to gUh, "Wy" to gWy
            ),
            mapOf("bz" to gbz, "br" to gbr, "bh" to gbh, "by" to gby),
            catIdx, gEmb, lr
        )
        return loss
    }

    @Suppress("UNCHECKED_CAST")
    private fun adamStep(
        matGrads: Map<String, Array<DoubleArray>>,
        vecGrads: Map<String, DoubleArray>,
        catIdx: Int,
        embGrad: DoubleArray,
        lr: Double,
        b1: Double = 0.9, b2: Double = 0.999, eps: Double = 1e-8, clip: Double = 5.0
    ) {
        if (adamM == null) {
            adamM = mutableMapOf(); adamV = mutableMapOf(); adamT = 0
            for (k in matGrads.keys) { adamM!![k] = zerosM(matGrads[k]!!.size, matGrads[k]!![0].size); adamV!![k] = zerosM(matGrads[k]!!.size, matGrads[k]!![0].size) }
            for (k in vecGrads.keys) { adamM!![k] = zeros(vecGrads[k]!!.size); adamV!![k] = zeros(vecGrads[k]!!.size) }
            adamM!!["Emb"] = zeros(embedDim); adamV!!["Emb"] = zeros(embedDim)
        }
        adamT += 1
        val bc1 = 1 - b1.pow(adamT); val bc2 = 1 - b2.pow(adamT)

        fun clipMat(g: Array<DoubleArray>): Array<DoubleArray> {
            var norm = 0.0
            for (row in g) for (v in row) norm += v * v
            norm = sqrt(norm)
            if (norm > clip) { val s = clip / (norm + 1e-8); for (row in g) for (j in row.indices) row[j] *= s }
            return g
        }
        fun clipVec(g: DoubleArray): DoubleArray {
            var norm = 0.0; for (v in g) norm += v * v; norm = sqrt(norm)
            if (norm > clip) { val s = clip / (norm + 1e-8); for (i in g.indices) g[i] *= s }
            return g
        }

        val params = mapOf(
            "Wz" to Wz, "Uz" to Uz, "Wr" to Wr, "Ur" to Ur, "Wh" to Wh, "Uh" to Uh, "Wy" to Wy
        )
        for ((k, g0) in matGrads) {
            val g = clipMat(g0)
            val m = adamM!![k] as Array<DoubleArray>; val v = adamV!![k] as Array<DoubleArray>
            val p = params[k]!!
            for (i in p.indices) for (j in p[i].indices) {
                m[i][j] = b1 * m[i][j] + (1 - b1) * g[i][j]
                v[i][j] = b2 * v[i][j] + (1 - b2) * g[i][j] * g[i][j]
                val mhat = m[i][j] / bc1; val vhat = v[i][j] / bc2
                p[i][j] -= lr * mhat / (sqrt(vhat) + eps)
            }
        }
        val vparams = mapOf("bz" to bz, "br" to br, "bh" to bh, "by" to by)
        for ((k, g0) in vecGrads) {
            val g = clipVec(g0)
            val m = adamM!![k] as DoubleArray; val v = adamV!![k] as DoubleArray
            val p = vparams[k]!!
            for (i in p.indices) {
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
                val mhat = m[i] / bc1; val vhat = v[i] / bc2
                p[i] -= lr * mhat / (sqrt(vhat) + eps)
            }
        }
        run {
            val g = clipVec(embGrad)
            val m = adamM!!["Emb"] as DoubleArray; val v = adamV!!["Emb"] as DoubleArray
            for (i in g.indices) {
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
                val mhat = m[i] / bc1; val vhat = v[i] / bc2
                Emb[catIdx][i] -= lr * mhat / (sqrt(vhat) + eps)
            }
        }
    }

    fun generate(catIdx: Int, nFrames: Int = 130, temperature: Double = 0.5, seed: Long = System.nanoTime()): Array<FloatArray> {
        val gRnd = Random(seed)
        val emb = Emb[catIdx]
        var h = DoubleArray(H)
        var prevFrame = DoubleArray(nMels)
        val out = Array(nFrames) { FloatArray(nMels) }
        for (t in 0 until nFrames) {
            val xt = concat(prevFrame, emb)
            val (hNew, y, _) = step(xt, h)
            h = hNew
            val noisy = DoubleArray(nMels) { y[it] + gaussian(gRnd) * temperature * 0.6 }
            out[t] = FloatArray(nMels) { noisy[it].toFloat() }
            prevFrame = noisy
        }
        return out
    }

    private fun gaussian(r: Random): Double {
        var u1 = r.nextDouble(); if (u1 < 1e-12) u1 = 1e-12
        val u2 = r.nextDouble()
        return sqrt(-2.0 * ln(u1)) * cos(2 * PI * u2)
    }

    // ---------------- binary persistence (no npz/numpy here, just raw doubles) ----

    fun writeTo(out: java.io.DataOutputStream) {
        out.writeInt(nMels)
        out.writeInt(hidden)
        out.writeInt(embedDim)
        out.writeInt(categories.size)
        for (c in categories) {
            val bytes = c.toByteArray(Charsets.UTF_8)
            out.writeInt(bytes.size)
            out.write(bytes)
        }
        fun writeMat(m: Array<DoubleArray>) { for (row in m) for (v in row) out.writeDouble(v) }
        fun writeVec(v: DoubleArray) { for (x in v) out.writeDouble(x) }

        writeMat(Wz); writeMat(Uz); writeVec(bz)
        writeMat(Wr); writeMat(Ur); writeVec(br)
        writeMat(Wh); writeMat(Uh); writeVec(bh)
        writeMat(Wy); writeVec(by)
        writeMat(Emb)
    }

    companion object {
        fun readFrom(input: java.io.DataInputStream): MelGru {
            val nMels = input.readInt()
            val hidden = input.readInt()
            val embedDim = input.readInt()
            val numCats = input.readInt()
            val cats = MutableList(numCats) {
                val len = input.readInt()
                val bytes = ByteArray(len)
                input.readFully(bytes)
                String(bytes, Charsets.UTF_8)
            }
            val m = MelGru(nMels = nMels, hidden = hidden, embedDim = embedDim, categories = cats)

            fun readMat(rows: Int, cols: Int): Array<DoubleArray> =
                Array(rows) { DoubleArray(cols) { input.readDouble() } }
            fun readVec(n: Int): DoubleArray = DoubleArray(n) { input.readDouble() }

            val inDim = nMels + embedDim
            m.Wz = readMat(hidden, inDim); m.Uz = readMat(hidden, hidden); m.bz = readVec(hidden)
            m.Wr = readMat(hidden, inDim); m.Ur = readMat(hidden, hidden); m.br = readVec(hidden)
            m.Wh = readMat(hidden, inDim); m.Uh = readMat(hidden, hidden); m.bh = readVec(hidden)
            m.Wy = readMat(nMels, hidden); m.by = readVec(nMels)
            m.Emb = readMat(maxOf(1, numCats), embedDim)
            return m
        }
    }
}

