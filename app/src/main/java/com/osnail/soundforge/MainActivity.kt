package com.osnail.soundforge

import android.app.AlertDialog
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.lifecycle.lifecycleScope
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var lib: Library
    private var model: NgramModel? = null
    private var lastPcm: FloatArray? = null
    private var pendingCategory: Category? = null

    private lateinit var inputPrompt: EditText
    private lateinit var seekVariation: SeekBar
    private lateinit var labelVariation: TextView
    private lateinit var txtResult: TextView
    private lateinit var txtModel: TextView
    private lateinit var txtStatus: TextView
    private lateinit var listCategories: LinearLayout

    // ---- file pickers -------------------------------------------------

    private val pickAudio = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        val cat = pendingCategory
        pendingCategory = null
        if (cat == null || uris.isNullOrEmpty()) return@registerForActivityResult
        importAudio(cat, uris)
    }

    private val exportPack = registerForActivityResult(
        ActivityResultContracts.CreateDocument("application/zip")
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        runIo {
            contentResolver.openOutputStream(uri)!!.use { DataPack.export(lib, it) }
        }
    }

    private val importPack = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        askMergeMode(uri)
    }

    private val saveWav = registerForActivityResult(
        ActivityResultContracts.CreateDocument("audio/wav")
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        val pcm = lastPcm
        if (pcm == null) {
            status("Nothing to save — generate a sound first")
            return@registerForActivityResult
        }
        runIo {
            contentResolver.openOutputStream(uri)!!.use { WavIO.write(it, pcm) }
            "Saved WAV"
        }
    }

    // ---- lifecycle ----------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputPrompt = findViewById(R.id.inputPrompt)
        seekVariation = findViewById(R.id.seekVariation)
        labelVariation = findViewById(R.id.labelVariation)
        txtResult = findViewById(R.id.txtResult)
        txtModel = findViewById(R.id.txtModel)
        txtStatus = findViewById(R.id.txtStatus)
        listCategories = findViewById(R.id.listCategories)

        lib = Library(this)
        lib.load()
        loadModel()

        seekVariation.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                labelVariation.text = "Variation: $p%"
            }

            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        findViewById<Button>(R.id.btnGenerate).setOnClickListener { generate() }
        findViewById<Button>(R.id.btnReplay).setOnClickListener {
            lastPcm?.let { Player.play(it) } ?: toast("Nothing to replay")
        }
        findViewById<Button>(R.id.btnSaveWav).setOnClickListener {
            if (lastPcm == null) toast("Generate something first")
            else saveWav.launch("soundforge_${System.currentTimeMillis()}.wav")
        }
        findViewById<Button>(R.id.btnTrain).setOnClickListener { train() }
        findViewById<Button>(R.id.btnAddCategory).setOnClickListener { addCategoryDialog() }
        findViewById<Button>(R.id.btnExport).setOnClickListener {
            exportPack.launch("soundforge_${System.currentTimeMillis()}.sfpack.zip")
        }
        findViewById<Button>(R.id.btnImport).setOnClickListener {
            importPack.launch(arrayOf("application/zip", "application/octet-stream", "*/*"))
        }

        renderCategories()
        updateModelInfo()
    }

    override fun onDestroy() {
        super.onDestroy()
        Player.stop()
    }

    // ---- model --------------------------------------------------------

    private fun loadModel() {
        model = try {
            if (lib.modelFile.exists()) NgramModel.fromJson(JSONObject(lib.modelFile.readText())) else null
        } catch (e: Exception) {
            null
        }
    }

    private fun train() {
        if (lib.categories.isEmpty()) {
            toast("Add a category first")
            return
        }
        val m = NgramModel(3)
        m.train(lib.categories)
        model = m
        updateModelInfo()
        try {
            lib.modelFile.writeText(m.toJson().toString())
            status("Model trained on ${m.trainedCategories} categories")
        } catch (e: Exception) {
            // The in-memory model is live either way; only persistence failed.
            status("Trained, but saving model.json failed: ${e.message}")
        }
    }

    private fun updateModelInfo() {
        val m = model
        txtModel.text = if (m == null || !m.isTrained) {
            "Not trained yet. Add categories + sounds, then hit TRAIN."
        } else {
            "Trained: ${m.trainedCategories} categories, ${m.vocabSize} n-grams (n=${m.n}) • " +
                "${lib.totalSamples()} samples, ${lib.storageBytes() / 1024} KB"
        }
    }

    // ---- generation ---------------------------------------------------

    private fun generate() {
        val prompt = inputPrompt.text.toString().trim()
        if (prompt.isEmpty()) { toast("Type a prompt"); return }
        val m = model
        if (m == null || !m.isTrained) { toast("Train the model first"); return }

        val matches = m.match(prompt)
        if (matches.isEmpty()) {
            txtResult.text = "No category matched \"$prompt\".\nTry adding an alias to a category."
            return
        }

        val variation = seekVariation.progress / 100f
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                Generator.generate(lib, matches, variation)
            }
            if (result.pcm.isEmpty()) {
                txtResult.text = result.log
                return@launch
            }
            lastPcm = result.pcm
            txtResult.text = result.log
            Player.play(result.pcm)
        }
    }

    // ---- categories ---------------------------------------------------

    private fun addCategoryDialog() {
        val name = EditText(this).apply {
            hint = "category name (e.g. boom)"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val aliases = EditText(this).apply {
            hint = "aliases, comma separated (ledakan, explosion, blast)"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 0)
            addView(name)
            addView(aliases)
        }
        AlertDialog.Builder(this)
            .setTitle("New category")
            .setView(box)
            .setPositiveButton("Add") { _, _ ->
                val cat = lib.addCategory(name.text.toString())
                if (cat == null) { toast("Invalid or duplicate name"); return@setPositiveButton }
                aliases.text.toString().split(",").map { it.trim() }.filter { it.isNotEmpty() }
                    .forEach { cat.aliases.add(it) }
                lib.save()
                renderCategories()
                status("Category \"${cat.name}\" added — remember to TRAIN again")
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun aliasDialog(cat: Category) {
        val field = EditText(this).apply {
            setText(cat.aliases.joinToString(", "))
            hint = "comma separated"
        }
        AlertDialog.Builder(this)
            .setTitle("Aliases for \"${cat.name}\"")
            .setView(field)
            .setPositiveButton("Save") { _, _ ->
                cat.aliases.clear()
                field.text.toString().split(",").map { it.trim() }.filter { it.isNotEmpty() }
                    .forEach { cat.aliases.add(it) }
                lib.save()
                renderCategories()
                status("Aliases updated — TRAIN again to apply")
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun renderCategories() {
        listCategories.removeAllViews()
        if (lib.categories.isEmpty()) {
            val tv = TextView(this).apply {
                text = "No categories yet. Add one, then load sounds into it."
                textSize = 12f
            }
            listCategories.addView(tv)
            return
        }
        for (cat in lib.categories) {
            val row = layoutInflater.inflate(R.layout.item_category, listCategories, false)
            row.findViewById<TextView>(R.id.txtCatName).text = cat.name
            row.findViewById<TextView>(R.id.txtCatInfo).text =
                "${cat.samples.size} sound(s)" +
                    if (cat.aliases.isEmpty()) "" else " • ${cat.aliases.joinToString(", ")}"

            row.findViewById<Button>(R.id.btnAddSound).setOnClickListener {
                pendingCategory = cat
                pickAudio.launch(arrayOf("audio/*"))
            }
            row.findViewById<Button>(R.id.btnAliases).setOnClickListener { aliasDialog(cat) }
            row.findViewById<Button>(R.id.btnPreview).setOnClickListener {
                if (cat.samples.isEmpty()) { toast("No sounds in this category"); return@setOnClickListener }
                val s = cat.samples.random()
                lifecycleScope.launch {
                    // A sample can be a couple of MB — never decode it on the UI thread.
                    val pcm = withContext(Dispatchers.IO) { lib.loadSample(s) }
                    if (pcm.isEmpty()) toast("Sample file is missing") else Player.play(pcm)
                }
            }
            row.findViewById<Button>(R.id.btnDelete).setOnClickListener {
                AlertDialog.Builder(this)
                    .setTitle("Delete \"${cat.name}\"?")
                    .setMessage("This removes the category and its ${cat.samples.size} sound file(s).")
                    .setPositiveButton("Delete") { _, _ ->
                        lib.deleteCategory(cat.name)
                        renderCategories()
                        updateModelInfo()
                        status("Deleted — TRAIN again")
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
            listCategories.addView(row)
        }
    }

    private fun importAudio(cat: Category, uris: List<Uri>) {
        status("Decoding ${uris.size} file(s)...")
        lifecycleScope.launch {
            val msg = withContext(Dispatchers.IO) {
                var ok = 0
                var fail = 0
                for (u in uris) {
                    try {
                        val pcm = AudioDecoder.decode(this@MainActivity, u)
                        if (pcm.isEmpty()) { fail++; continue }
                        val label = u.lastPathSegment?.substringAfterLast('/') ?: "sound"
                        lib.addSample(cat, pcm, label)
                        ok++
                    } catch (e: Exception) {
                        fail++
                    }
                }
                "Added $ok sound(s) to \"${cat.name}\"" + if (fail > 0) ", $fail failed" else ""
            }
            renderCategories()
            updateModelInfo()
            status("$msg — TRAIN again to update the model")
        }
    }

    // ---- data pack ----------------------------------------------------

    private fun askMergeMode(uri: Uri) {
        AlertDialog.Builder(this)
            .setTitle("Import pack")
            .setMessage("Merge into your current library, or replace everything?")
            .setPositiveButton("Merge") { _, _ -> doImport(uri, true) }
            .setNegativeButton("Replace") { _, _ -> doImport(uri, false) }
            .setNeutralButton("Cancel", null)
            .show()
    }

    private fun doImport(uri: Uri, merge: Boolean) {
        status("Importing...")
        lifecycleScope.launch {
            val msg = withContext(Dispatchers.IO) {
                try {
                    contentResolver.openInputStream(uri)!!.use { DataPack.import(lib, it, merge) }
                } catch (e: Exception) {
                    "Import failed: ${e.message}"
                }
            }
            lib.load()
            loadModel()
            renderCategories()
            updateModelInfo()
            status("$msg — TRAIN again if matching feels off")
        }
    }

    // ---- helpers ------------------------------------------------------

    private fun runIo(block: suspend () -> Any?) {
        lifecycleScope.launch {
            val msg = withContext(Dispatchers.IO) {
                try {
                    block()?.toString() ?: "Done"
                } catch (e: Exception) {
                    "Failed: ${e.message}"
                }
            }
            status(msg)
        }
    }

    private fun status(msg: String) {
        txtStatus.text = msg
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }
}
