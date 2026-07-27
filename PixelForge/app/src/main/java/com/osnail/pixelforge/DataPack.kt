package com.osnail.pixelforge

import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * A .pxpack file is a zip:
 *   library.json
 *   matcher.json (n-gram text matcher, optional)
 *   model/pixelgru.bin (trained generative model, optional)
 *   model/train_meta.json (optional)
 *   images/ (one .pix per drawing, raw 256-byte palette-index files)
 */
object DataPack {

    fun export(lib: Library, out: OutputStream): String {
        var files = 0
        ZipOutputStream(out.buffered()).use { zip ->
            if (lib.libFile.exists()) {
                zip.putNextEntry(ZipEntry("library.json")); zip.write(lib.libFile.readBytes()); zip.closeEntry()
            }
            if (lib.matcherFile.exists()) {
                zip.putNextEntry(ZipEntry("matcher.json")); zip.write(lib.matcherFile.readBytes()); zip.closeEntry()
            }
            val gruFile = Trainer.modelFile(lib)
            if (gruFile.exists()) {
                zip.putNextEntry(ZipEntry("model/pixelgru.bin")); zip.write(gruFile.readBytes()); zip.closeEntry()
            }
            val metaFile = File(lib.modelDir, "train_meta.json")
            if (metaFile.exists()) {
                zip.putNextEntry(ZipEntry("model/train_meta.json")); zip.write(metaFile.readBytes()); zip.closeEntry()
            }
            for (cat in lib.categories) {
                for (s in cat.samples) {
                    val f = File(lib.imagesDir, s.file)
                    if (!f.exists()) continue
                    zip.putNextEntry(ZipEntry("images/${s.file}"))
                    f.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                    files++
                }
            }
        }
        return "Exported ${lib.categories.size} categories, $files drawings"
    }

    fun import(lib: Library, input: InputStream, merge: Boolean): String {
        val tmpDir = File(lib.imagesDir.parentFile, "import_tmp")
        tmpDir.deleteRecursively()
        tmpDir.mkdirs()
        val tmpImages = File(tmpDir, "images").apply { mkdirs() }

        var libJson: ByteArray? = null
        var matcherJson: ByteArray? = null
        var gruBytes: ByteArray? = null
        var trainMetaBytes: ByteArray? = null

        ZipInputStream(input.buffered()).use { zip ->
            var entry: ZipEntry? = zip.nextEntry
            while (entry != null) {
                val name = entry.name.replace('\\', '/')
                when {
                    name.endsWith("library.json") -> libJson = zip.readBytes()
                    name.endsWith("matcher.json") -> matcherJson = zip.readBytes()
                    name.endsWith("pixelgru.bin") -> gruBytes = zip.readBytes()
                    name.endsWith("train_meta.json") -> trainMetaBytes = zip.readBytes()
                    name.contains("images/") && name.endsWith(".pix") -> {
                        val safe = name.substringAfterLast('/')
                        if (safe.isNotEmpty()) File(tmpImages, safe).outputStream().use { zip.copyTo(it) }
                    }
                }
                zip.closeEntry()
                entry = zip.nextEntry
            }
        }

        if (libJson == null) {
            tmpDir.deleteRecursively()
            return "Invalid pack: library.json missing"
        }

        val parsed = parseCategories(String(libJson!!, Charsets.UTF_8))

        if (!merge) {
            lib.imagesDir.listFiles()?.forEach { it.delete() }
            lib.categories.clear()
            if (gruBytes == null) {
                Trainer.modelFile(lib).delete()
                File(lib.modelDir, "train_meta.json").delete()
            }
        }

        var added = 0
        for (cat in parsed) {
            val target = lib.find(cat.name) ?: Category(cat.name).also { lib.categories.add(it) }
            for (a in cat.aliases) if (!target.aliases.contains(a)) target.aliases.add(a)
            for (s in cat.samples) {
                val src = File(tmpImages, s.file)
                if (!src.exists()) continue
                var destName = s.file
                if (target.samples.any { it.file == destName }) {
                    destName = "i_${System.currentTimeMillis()}_${(0..9999).random()}.pix"
                }
                src.copyTo(File(lib.imagesDir, destName), overwrite = true)
                target.samples.add(Sample(destName, s.label))
                added++
            }
        }

        lib.save()
        matcherJson?.let { lib.matcherFile.writeBytes(it) }
        gruBytes?.let { Trainer.modelFile(lib).writeBytes(it) }
        trainMetaBytes?.let { File(lib.modelDir, "train_meta.json").writeBytes(it) }
        tmpDir.deleteRecursively()
        val gruNote = if (gruBytes != null) ", trained model included" else ""
        return "Imported ${parsed.size} categories, $added drawings$gruNote"
    }

    private fun parseCategories(json: String): List<Category> {
        val out = ArrayList<Category>()
        try {
            val root = org.json.JSONObject(json)
            val arr = root.optJSONArray("categories") ?: return out
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val c = Category(o.getString("name"))
                val al = o.optJSONArray("aliases")
                if (al != null) for (j in 0 until al.length()) c.aliases.add(al.getString(j))
                val sa = o.optJSONArray("samples")
                if (sa != null) for (j in 0 until sa.length()) {
                    val s = sa.getJSONObject(j)
                    c.samples.add(Sample(s.getString("file"), s.optString("label", "")))
                }
                out.add(c)
            }
        } catch (e: Exception) {
        }
        return out
    }
}
