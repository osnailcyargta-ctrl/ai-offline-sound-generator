package com.osnail.pixelforge

import kotlin.math.*
import kotlin.random.Random

class PixelGru(
    val paletteSize: Int = 24,
    val hidden: Int = 96,
    val colorEmbedDim: Int = 8,
    val embedDim: Int = 16,
    val gridSize: Int = 16,
    var categories: MutableList<String> = mutableListOf(),
    seed: Long = 0
) {
    private val rnd = Random(seed)
    val posDim = gridSize
    val inDim = colorEmbedDim * 2 + posDim * 2 + embedDim
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
    var Wy = init(paletteSize, H); var by = zeros(paletteSize)
    var ColorEmb = Array(paletteSize) { DoubleArray(colorEmbedDim) { (rnd.nextDouble() * 2 - 1) * 0.15 } }
    var Emb = Array(max(1, categories.size)) { DoubleArray(embedDim) { (rnd.nextDouble() * 2 - 1) * 0.1 } }

    private var adamM: MutableMap<String, Any>? = null
    private var adamV: MutableMap<String, Any>? = null
    private var adamT = 0

    fun categoryIndex(name: String): Int { val i = categories.indexOf(name); return if (i < 0) 0 else i }

    private fun matVec(m: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        val out = DoubleArray(m.size)
        for (i in m.indices) { var s = 0.0; val row = m[i]; for (j in v.indices) s += row[j] * v[j]; out[i] = s }
        return out
    }
    private fun matVecT(m: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        val cols = m[0].size; val out = DoubleArray(cols)
        for (i in m.indices) { val vi = v[i]; if (vi == 0.0) continue; val row = m[i]; for (j in 0 until cols) out[j] += row[j] * vi }
        return out
    }
    private fun add3(a: DoubleArray, b: DoubleArray, c: DoubleArray) = DoubleArray(a.size) { a[it] + b[it] + c[it] }
    private fun mulElem(a: DoubleArray, b: DoubleArray) = DoubleArray(a.size) { a[it] * b[it] }
    private fun sigmoid(x: DoubleArray) = DoubleArray(x.size) { 1.0 / (1.0 + exp(-x[it].coerceIn(-30.0, 30.0))) }
    private fun tanhV(x: DoubleArray) = DoubleArray(x.size) { tanh(x[it]) }
    private fun outer(a: DoubleArray, b: DoubleArray): Array<DoubleArray> = Array(a.size) { i -> DoubleArray(b.size) { j -> a[i] * b[j] } }
    private fun addInPlace(m: Array<DoubleArray>, d: Array<DoubleArray>) { for (i in m.indices) for (j in m[i].indices) m[i][j] += d[i][j] }
    private fun addInPlaceV(v: DoubleArray, d: DoubleArray) { for (i in v.indices) v[i] += d[i] }
    private fun oneHot(idx: Int, n: Int) = DoubleArray(n) { if (it == idx) 1.0 else 0.0 }

    private fun buildInput(leftIdx: Int, aboveIdx: Int, row: Int, col: Int, emb: DoubleArray): DoubleArray {
        val out = DoubleArray(inDim)
        var p = 0
        for (v in ColorEmb[leftIdx]) out[p++] = v
        for (v in ColorEmb[aboveIdx]) out[p++] = v
        for (i in 0 until posDim) out[p++] = if (i == row) 1.0 else 0.0
        for (i in 0 until posDim) out[p++] = if (i == col) 1.0 else 0.0
        for (v in emb) out[p++] = v
        return out
    }

    data class GateCache(val z: DoubleArray, val r: DoubleArray, val hTilde: DoubleArray)

    private fun step(xt: DoubleArray, hPrev: DoubleArray): Triple<DoubleArray, DoubleArray, GateCache> {
        val z = sigmoid(add3(matVec(Wz, xt), matVec(Uz, hPrev), bz))
        val r = sigmoid(add3(matVec(Wr, xt), matVec(Ur, hPrev), br))
        val rh = mulElem(r, hPrev)
        val hTilde = tanhV(add3(matVec(Wh, xt), matVec(Uh, rh), bh))
        val h = DoubleArray(H) { (1 - z[it]) * hPrev[it] + z[it] * hTilde[it] }
        val logits = DoubleArray(paletteSize)
        val wy = matVec(Wy, h)
        for (i in 0 until paletteSize) logits[i] = wy[i] + by[i]
        return Triple(h, logits, GateCache(z, r, hTilde))
    }

    private fun softmax(logits: DoubleArray): DoubleArray {
        var mx = logits[0]; for (v in logits) if (v > mx) mx = v
        val exps = DoubleArray(logits.size) { exp(logits[it] - mx) }
        var sum = 0.0; for (v in exps) sum += v
        return DoubleArray(logits.size) { exps[it] / sum }
    }

    /** pixels: length gridSize*gridSize, values are palette indices (row-major). */
    fun trainSequence(pixels: IntArray, catIdx: Int, lr: Double = 0.01): Double {
        val n = gridSize * gridSize
        val emb = Emb[catIdx]

        val hs = ArrayList<DoubleArray>(n + 1)
        val probsList = ArrayList<DoubleArray>(n)
        val caches = ArrayList<GateCache>(n)
        val xs = ArrayList<DoubleArray>(n)
        val leftIdxs = IntArray(n); val aboveIdxs = IntArray(n)

        var h = DoubleArray(H)
        hs.add(h)
        var loss = 0.0
        for (t in 0 until n) {
            val row = t / gridSize; val col = t % gridSize
            val leftIdx = if (col == 0) 0 else pixels[t - 1]
            val aboveIdx = if (row == 0) 0 else pixels[t - gridSize]
            leftIdxs[t] = leftIdx; aboveIdxs[t] = aboveIdx

            val xt = buildInput(leftIdx, aboveIdx, row, col, emb)
            val (hNew, logits, cache) = step(xt, h)
            h = hNew
            val probs = softmax(logits)
            probsList.add(probs); caches.add(cache); xs.add(xt); hs.add(h)

            val target = pixels[t]
            loss += -ln(probs[target].coerceAtLeast(1e-9))
        }
        loss /= n

        val gWz = zerosM(H, inDim); val gUz = zerosM(H, H); val gbz = zeros(H)
        val gWr = zerosM(H, inDim); val gUr = zerosM(H, H); val gbr = zeros(H)
        val gWh = zerosM(H, inDim); val gUh = zerosM(H, H); val gbh = zeros(H)
        val gWy = zerosM(paletteSize, H); val gby = zeros(paletteSize)
        val gEmb = zeros(embedDim)
        val gColorEmb = zerosM(paletteSize, colorEmbedDim)

        var dHNext = DoubleArray(H)
        for (t in n - 1 downTo 0) {
            val cache = caches[t]; val z = cache.z; val r = cache.r; val hTilde = cache.hTilde
            val hPrev = hs[t]; val hT = hs[t + 1]; val xt = xs[t]
            val probs = probsList[t]; val target = pixels[t]

            val dLogits = DoubleArray(paletteSize) { (probs[it] - (if (it == target) 1.0 else 0.0)) / n }
            addInPlace(gWy, outer(dLogits, hT))
            addInPlaceV(gby, dLogits)

            val dH2 = DoubleArray(H)
            for (i in 0 until paletteSize) { val d = dLogits[i]; if (d == 0.0) continue; val row = Wy[i]; for (j in 0 until H) dH2[j] += row[j] * d }
            for (j in 0 until H) dH2[j] += dHNext[j]

            val dZ = DoubleArray(H) { dH2[it] * (hTilde[it] - hPrev[it]) }
            val dHTilde = DoubleArray(H) { dH2[it] * z[it] }
            val dHPrevDirect = DoubleArray(H) { dH2[it] * (1 - z[it]) }

            val dHTildeRaw = DoubleArray(H) { dHTilde[it] * (1 - hTilde[it] * hTilde[it]) }
            addInPlace(gWh, outer(dHTildeRaw, xt))
            val rh = mulElem(r, hPrev)
            addInPlace(gUh, outer(dHTildeRaw, rh))
            addInPlaceV(gbh, dHTildeRaw)

            val dRH = matVecT(Uh, dHTildeRaw)
            val dR = DoubleArray(H) { dRH[it] * hPrev[it] }
            val dRHprevPart = DoubleArray(H) { dRH[it] * r[it] }

            val dZRaw = DoubleArray(H) { dZ[it] * z[it] * (1 - z[it]) }
            addInPlace(gWz, outer(dZRaw, xt)); addInPlace(gUz, outer(dZRaw, hPrev)); addInPlaceV(gbz, dZRaw)

            val dRRaw = DoubleArray(H) { dR[it] * r[it] * (1 - r[it]) }
            addInPlace(gWr, outer(dRRaw, xt)); addInPlace(gUr, outer(dRRaw, hPrev)); addInPlaceV(gbr, dRRaw)

            val dXfromZ = matVecT(Wz, dZRaw)
            val dXfromR = matVecT(Wr, dRRaw)
            val dXfromH = matVecT(Wh, dHTildeRaw)
            val dX = DoubleArray(inDim) { dXfromZ[it] + dXfromR[it] + dXfromH[it] }

            // unpack dX back into leftColorEmb / aboveColorEmb / (pos has no grad) / category emb
            var p = 0
            for (k in 0 until colorEmbedDim) gColorEmb[leftIdxs[t]][k] += dX[p++]
            for (k in 0 until colorEmbedDim) gColorEmb[aboveIdxs[t]][k] += dX[p++]
            p += posDim * 2 // position one-hot: no learnable params
            for (k in 0 until embedDim) gEmb[k] += dX[p++]

            val dUzT = matVecT(Uz, dZRaw); val dUrT = matVecT(Ur, dRRaw)
            dHNext = DoubleArray(H) { dHPrevDirect[it] + dUzT[it] + dUrT[it] + dRHprevPart[it] }
        }

        adamStep(
            mapOf("Wz" to gWz, "Uz" to gUz, "Wr" to gWr, "Ur" to gUr, "Wh" to gWh, "Uh" to gUh, "Wy" to gWy),
            mapOf("bz" to gbz, "br" to gbr, "bh" to gbh, "by" to gby),
            catIdx, gEmb, gColorEmb, lr
        )
        return loss
    }

    @Suppress("UNCHECKED_CAST")
    private fun adamStep(
        matGrads: Map<String, Array<DoubleArray>>, vecGrads: Map<String, DoubleArray>,
        catIdx: Int, embGrad: DoubleArray, colorEmbGrad: Array<DoubleArray>, lr: Double,
        b1: Double = 0.9, b2: Double = 0.999, eps: Double = 1e-8, clip: Double = 5.0
    ) {
        if (adamM == null) {
            adamM = mutableMapOf(); adamV = mutableMapOf(); adamT = 0
            for (k in matGrads.keys) { adamM!![k] = zerosM(matGrads[k]!!.size, matGrads[k]!![0].size); adamV!![k] = zerosM(matGrads[k]!!.size, matGrads[k]!![0].size) }
            for (k in vecGrads.keys) { adamM!![k] = zeros(vecGrads[k]!!.size); adamV!![k] = zeros(vecGrads[k]!!.size) }
            adamM!!["Emb"] = zeros(embedDim); adamV!!["Emb"] = zeros(embedDim)
            adamM!!["ColorEmb"] = zerosM(paletteSize, colorEmbedDim); adamV!!["ColorEmb"] = zerosM(paletteSize, colorEmbedDim)
        }
        adamT += 1
        val bc1 = 1 - b1.pow(adamT); val bc2 = 1 - b2.pow(adamT)
        fun clipMat(g: Array<DoubleArray>): Array<DoubleArray> { var n=0.0; for(row in g) for(v in row) n+=v*v; n=sqrt(n); if(n>clip){val s=clip/(n+1e-8); for(row in g) for(j in row.indices) row[j]*=s}; return g }
        fun clipVec(g: DoubleArray): DoubleArray { var n=0.0; for(v in g) n+=v*v; n=sqrt(n); if(n>clip){val s=clip/(n+1e-8); for(i in g.indices) g[i]*=s}; return g }

        val params = mapOf("Wz" to Wz, "Uz" to Uz, "Wr" to Wr, "Ur" to Ur, "Wh" to Wh, "Uh" to Uh, "Wy" to Wy)
        for ((k, g0) in matGrads) {
            val g = clipMat(g0); val m = adamM!![k] as Array<DoubleArray>; val v = adamV!![k] as Array<DoubleArray>; val p = params[k]!!
            for (i in p.indices) for (j in p[i].indices) {
                m[i][j] = b1*m[i][j]+(1-b1)*g[i][j]; v[i][j] = b2*v[i][j]+(1-b2)*g[i][j]*g[i][j]
                p[i][j] -= lr * (m[i][j]/bc1) / (sqrt(v[i][j]/bc2)+eps)
            }
        }
        val vparams = mapOf("bz" to bz, "br" to br, "bh" to bh, "by" to by)
        for ((k, g0) in vecGrads) {
            val g = clipVec(g0); val m = adamM!![k] as DoubleArray; val v = adamV!![k] as DoubleArray; val p = vparams[k]!!
            for (i in p.indices) { m[i]=b1*m[i]+(1-b1)*g[i]; v[i]=b2*v[i]+(1-b2)*g[i]*g[i]; p[i] -= lr*(m[i]/bc1)/(sqrt(v[i]/bc2)+eps) }
        }
        run {
            val g = clipVec(embGrad); val m = adamM!!["Emb"] as DoubleArray; val v = adamV!!["Emb"] as DoubleArray
            for (i in g.indices) { m[i]=b1*m[i]+(1-b1)*g[i]; v[i]=b2*v[i]+(1-b2)*g[i]*g[i]; Emb[catIdx][i] -= lr*(m[i]/bc1)/(sqrt(v[i]/bc2)+eps) }
        }
        run {
            val g = clipMat(colorEmbGrad); val m = adamM!!["ColorEmb"] as Array<DoubleArray>; val v = adamV!!["ColorEmb"] as Array<DoubleArray>
            for (i in g.indices) for (j in g[i].indices) { m[i][j]=b1*m[i][j]+(1-b1)*g[i][j]; v[i][j]=b2*v[i][j]+(1-b2)*g[i][j]*g[i][j]; ColorEmb[i][j] -= lr*(m[i][j]/bc1)/(sqrt(v[i][j]/bc2)+eps) }
        }
    }

    fun generate(catIdx: Int, temperature: Double = 0.7, seed: Long = System.nanoTime()): IntArray =
        generateFromEmbedding(Emb[catIdx], temperature, seed)

    /** Blends multiple categories' embeddings (weighted) into one generation pass --
     * a genuine model-level concept blend, not a pixel-level cross-fade of two images. */
    fun generateBlended(catIdxs: List<Int>, weights: List<Double>, temperature: Double = 0.7, seed: Long = System.nanoTime()): IntArray {
        val wSum = weights.sum().coerceAtLeast(1e-6)
        val blended = DoubleArray(embedDim)
        for (i in catIdxs.indices) {
            val w = weights[i] / wSum
            val e = Emb[catIdxs[i]]
            for (k in 0 until embedDim) blended[k] += e[k] * w
        }
        return generateFromEmbedding(blended, temperature, seed)
    }

    private fun generateFromEmbedding(emb: DoubleArray, temperature: Double, seed: Long): IntArray {
        val gRnd = Random(seed)
        val n = gridSize * gridSize
        val pixels = IntArray(n)
        var h = DoubleArray(H)
        for (t in 0 until n) {
            val row = t / gridSize; val col = t % gridSize
            val leftIdx = if (col == 0) 0 else pixels[t - 1]
            val aboveIdx = if (row == 0) 0 else pixels[t - gridSize]
            val xt = buildInput(leftIdx, aboveIdx, row, col, emb)
            val (hNew, logits, _) = step(xt, h)
            h = hNew
            val scaled = DoubleArray(paletteSize) { logits[it] / temperature }
            val probs = softmax(scaled)
            var r = gRnd.nextDouble(); var cum = 0.0; var chosen = paletteSize - 1
            for (i in 0 until paletteSize) { cum += probs[i]; if (r <= cum) { chosen = i; break } }
            pixels[t] = chosen
        }
        return pixels
    }

    // ---------------- binary persistence ----------------

    fun writeTo(out: java.io.DataOutputStream) {
        out.writeInt(paletteSize)
        out.writeInt(hidden)
        out.writeInt(colorEmbedDim)
        out.writeInt(embedDim)
        out.writeInt(gridSize)
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
        writeMat(ColorEmb)
        writeMat(Emb)
    }

    companion object {
        fun readFrom(input: java.io.DataInputStream): PixelGru {
            val paletteSize = input.readInt()
            val hidden = input.readInt()
            val colorEmbedDim = input.readInt()
            val embedDim = input.readInt()
            val gridSize = input.readInt()
            val numCats = input.readInt()
            val cats = MutableList(numCats) {
                val len = input.readInt()
                val bytes = ByteArray(len)
                input.readFully(bytes)
                String(bytes, Charsets.UTF_8)
            }
            val m = PixelGru(
                paletteSize = paletteSize, hidden = hidden, colorEmbedDim = colorEmbedDim,
                embedDim = embedDim, gridSize = gridSize, categories = cats
            )

            fun readMat(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) { input.readDouble() } }
            fun readVec(n: Int): DoubleArray = DoubleArray(n) { input.readDouble() }

            val inDim = colorEmbedDim * 2 + gridSize * 2 + embedDim
            m.Wz = readMat(hidden, inDim); m.Uz = readMat(hidden, hidden); m.bz = readVec(hidden)
            m.Wr = readMat(hidden, inDim); m.Ur = readMat(hidden, hidden); m.br = readVec(hidden)
            m.Wh = readMat(hidden, inDim); m.Uh = readMat(hidden, hidden); m.bh = readVec(hidden)
            m.Wy = readMat(paletteSize, hidden); m.by = readVec(paletteSize)
            m.ColorEmb = readMat(paletteSize, colorEmbedDim)
            m.Emb = readMat(maxOf(1, numCats), embedDim)
            return m
        }
    }
}

