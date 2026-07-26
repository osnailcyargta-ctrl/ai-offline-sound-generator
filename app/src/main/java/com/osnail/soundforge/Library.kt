package com.osnail.soundforge

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class Sample(val file: String, val label: String, val frames: Int)

data class Category(
    val name: String,
    val aliases: MutableList<String> = mutableListOf(),
    val samples: MutableList<Sample> = mutableListOf()
)

class Library(private val ctx: Context) {

    val soundsDir: File = File(ctx.filesDir, "sounds").apply { mkdirs() }
    val libFile: File = File(ctx.filesDir, "library.json")
    val modelFile: File = File(ctx.filesDir, "model.json")
    val modelDir: File = File(ctx.filesDir, "model").apply { mkdirs() }

    val categories = mutableListOf<Category>()

    fun load() {
        categories.clear()
        if (!libFile.exists()) return
        try {
            val root = JSONObject(libFile.readText())
            val arr = root.optJSONArray("categories") ?: JSONArray()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val cat = Category(o.getString("name"))
                val al = o.optJSONArray("aliases") ?: JSONArray()
                for (j in 0 until al.length()) cat.aliases.add(al.getString(j))
                val sa = o.optJSONArray("samples") ?: JSONArray()
                for (j in 0 until sa.length()) {
                    val s = sa.getJSONObject(j)
                    cat.samples.add(Sample(s.getString("file"), s.optString("label", ""), s.optInt("frames", 0)))
                }
                categories.add(cat)
            }
        } catch (e: Exception) {
            categories.clear()
        }
    }

    fun save() {
        val root = JSONObject()
        val arr = JSONArray()
        for (c in categories) {
            val o = JSONObject()
            o.put("name", c.name)
            o.put("aliases", JSONArray(c.aliases))
            val sa = JSONArray()
            for (s in c.samples) {
                sa.put(JSONObject().put("file", s.file).put("label", s.label).put("frames", s.frames))
            }
            o.put("samples", sa)
            arr.put(o)
        }
        root.put("version", 1)
        root.put("categories", arr)
        libFile.writeText(root.toString())
    }

    fun find(name: String): Category? = categories.firstOrNull { it.name.equals(name, true) }

    fun addCategory(name: String): Category? {
        val clean = name.trim()
        if (clean.isEmpty() || find(clean) != null) return null
        val c = Category(clean)
        categories.add(c)
        save()
        return c
    }

    fun deleteCategory(name: String) {
        val c = find(name) ?: return
        for (s in c.samples) File(soundsDir, s.file).delete()
        categories.remove(c)
        save()
    }

    fun addSample(cat: Category, pcm: FloatArray, label: String) {
        val trimmed = Dsp.trimSilence(pcm)
        val fname = "s_${System.currentTimeMillis()}_${(0..9999).random()}.wav"
        WavIO.write(File(soundsDir, fname), trimmed)
        cat.samples.add(Sample(fname, label, trimmed.size))
        save()
    }

    fun loadSample(s: Sample): FloatArray {
        val f = File(soundsDir, s.file)
        return if (f.exists()) WavIO.read(f) else FloatArray(0)
    }

    fun totalSamples(): Int = categories.sumOf { it.samples.size }

    fun storageBytes(): Long = soundsDir.listFiles()?.sumOf { it.length() } ?: 0L

    fun saveTrainMeta(categories: List<String>, samplesUsed: Int, finalLoss: Double) {
        val o = JSONObject()
        o.put("categories", JSONArray(categories))
        o.put("samples_used", samplesUsed)
        o.put("final_loss", finalLoss)
        o.put("trained_at", System.currentTimeMillis())
        File(modelDir, "train_meta.json").writeText(o.toString())
    }

    fun trainMeta(): JSONObject? {
        val f = File(modelDir, "train_meta.json")
        if (!f.exists()) return null
        return try { JSONObject(f.readText()) } catch (e: Exception) { null }
    }
}
