package com.osnail.soundforge

import android.app.AlertDialog
import android.net.Uri
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import org.json.JSONObject
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var lib: Library
    private var matcher: NgramModel? = null
    private var melModel: MelGru? = null
    private var lastPcm: FloatArray? = null
    private var pendingCategory: Category? = null

    private lateinit var inputPrompt: EditText
    private lateinit var seekVariation: SeekBar
    private lateinit var labelVariation: TextView
    private lateinit var txtResult: TextView
    private lateinit var txtModel: TextView
    private lateinit var txtStatus: TextView
    private lateinit var listCategories: LinearLayout
    private lateinit var trainProgress: android.widget.ProgressBar
    private lateinit var txtTrainProgress: TextView
    private lateinit var txtRam: TextView
    private lateinit var labelDuration: TextView
    private lateinit var seekDuration: SeekBar
    private lateinit var inputDuration: EditText
    private lateinit var txtDurationCost: TextView
    private lateinit var labelEpochs: TextView
    private lateinit var seekEpochs: SeekBar

    /** Guards the two-way slider <-> text field sync from looping. */
    private var syncingDuration = false

    private companion object {
        const val MIN_SECONDS = 0.10f

        /**
         * Typing a custom length is allowed to go past the slider's 10s, but a
         * Griffin-Lim pass costs roughly 3.4 MB per second per layer, so an
         * unbounded value is a straight OOM. 60s is already ~200 MB.
         */
        const val MAX_SECONDS = 60.0f

        const val MIN_EPOCHS = 10
    }

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
        trainProgress = findViewById(R.id.trainProgress)
        txtTrainProgress = findViewById(R.id.txtTrainProgress)
        txtRam = findViewById(R.id.txtRam)
        labelDuration = findViewById(R.id.labelDuration)
        seekDuration = findViewById(R.id.seekDuration)
        inputDuration = findViewById(R.id.inputDuration)
        txtDurationCost = findViewById(R.id.txtDurationCost)
        labelEpochs = findViewById(R.id.labelEpochs)
        seekEpochs = findViewById(R.id.seekEpochs)

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

        setupDurationControls()
        setupEpochControl()
        startRamMonitor()

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

    // ---- duration / epochs / RAM --------------------------------------

    /** Slider position as seconds: progress 0..990 -> 0.10..10.00s. */
    private fun sliderSeconds(): Float = (seekDuration.progress + 10) / 100f

    /** What GENERATE actually uses: the typed value if it parses, else the slider. */
    private fun currentSeconds(): Float {
        val typed = inputDuration.text.toString().trim().toFloatOrNull()
        return (typed ?: sliderSeconds()).coerceIn(MIN_SECONDS, MAX_SECONDS)
    }

    private fun currentEpochs(): Int = MIN_EPOCHS + seekEpochs.progress

    /** Peak Griffin-Lim working set for one layer of [seconds], same shape the vocoder allocates. */
    private fun estimateMb(seconds: Float): Float {
        val frames = (seconds * WavIO.SAMPLE_RATE / MelFeatures.HOP).toInt().coerceAtLeast(8)
        val freqBins = MelFeatures.N_FFT / 2 + 1
        val spec = freqBins.toLong() * frames * 8L          // magFull + re + im
        val outLen = MelFeatures.N_FFT + MelFeatures.HOP.toLong() * (frames - 1)
        val scratch = outLen * 8L * 3L + (outLen - MelFeatures.N_FFT) * 4L * 2L
        return (3L * spec + scratch) / (1024f * 1024f)
    }

    private fun setupDurationControls() {
        seekDuration.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                if (fromUser) {
                    // Dragging wins over whatever was typed before.
                    syncingDuration = true
                    inputDuration.setText(String.format(Locale.US, "%.2f", (p + 10) / 100f))
                    syncingDuration = false
                }
                updateDurationInfo()
            }

            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        inputDuration.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                if (!syncingDuration) {
                    val v = s?.toString()?.trim()?.toFloatOrNull()
                    // Only follow the text while it is still on the slider's scale;
                    // past 10s the slider just parks at its maximum.
                    if (v != null && v >= MIN_SECONDS && v <= 10.0f) {
                        syncingDuration = true
                        seekDuration.progress = (v * 100).toInt() - 10
                        syncingDuration = false
                    }
                }
                updateDurationInfo()
            }
        })

        inputDuration.setText(String.format(Locale.US, "%.2f", sliderSeconds()))
        updateDurationInfo()
    }

    private fun updateDurationInfo() {
        val raw = inputDuration.text.toString().trim()
        val typed = raw.toFloatOrNull()
        val secs = (typed ?: sliderSeconds()).coerceIn(MIN_SECONDS, MAX_SECONDS)

        labelDuration.text = "Duration: ${String.format(Locale.US, "%.2f", secs)}s per layer"
        val note = when {
            raw.isNotEmpty() && typed == null -> " · not a number, using slider"
            typed != null && typed > MAX_SECONDS -> " · capped at ${MAX_SECONDS.toInt()}s"
            typed != null && typed < MIN_SECONDS -> " · min ${MIN_SECONDS}s"
            else -> ""
        }
        txtDurationCost.text = String.format(Locale.US, "~%.1f MB/layer%s", estimateMb(secs), note)
    }

    private fun setupEpochControl() {
        seekEpochs.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                labelEpochs.text = "Epochs: ${MIN_EPOCHS + p}"
            }

            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })
        labelEpochs.text = "Epochs: ${currentEpochs()}"
    }

    private fun startRamMonitor() {
        lifecycleScope.launch {
            // Only polls while the activity is actually on screen.
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                while (true) {
                    txtRam.text = ramLine()
                    delay(500)
                }
            }
        }
    }

    private fun ramLine(): String {
        val rt = Runtime.getRuntime()
        val mb = 1024.0 * 1024.0
        val used = (rt.totalMemory() - rt.freeMemory()) / mb
        val reserved = rt.totalMemory() / mb
        val limit = rt.maxMemory() / mb
        val native = android.os.Debug.getNativeHeapAllocatedSize() / mb
        return String.format(
            Locale.US,
            "RAM  heap %.1f MB used · %.0f reserved · %.0f limit · native %.1f MB",
            used, reserved, limit, native
        )
    }

    // ---- model --------------------------------------------------------

    private fun loadModel() {
        matcher = try {
            if (lib.modelFile.exists()) NgramModel.fromJson(JSONObject(lib.modelFile.readText())) else null
        } catch (e: Exception) {
            null
        }
        melModel = Trainer.loadModel(lib)
    }

    private fun train() {
        if (lib.categories.isEmpty()) {
            toast("Add a category first")
            return
        }
        if (Trainer.isTraining()) {
            toast("Training already running")
            return
        }

        // instant text-matcher rebuild (cheap, no need to wait for GRU training)
        val m = NgramModel(3)
        m.train(lib.categories)
        matcher = m
        try {
            lib.modelFile.writeText(m.toJson().toString())
        } catch (e: Exception) {
            // The matcher is live in memory either way; only persistence failed.
            status("Matcher saved failed: ${e.message}")
        }

        trainProgress.visibility = android.view.View.VISIBLE
        txtTrainProgress.visibility = android.view.View.VISIBLE
        findViewById<Button>(R.id.btnTrain).isEnabled = false
        // The count is captured at launch; moving the slider mid-run must not
        // change the epoch total the loop is already counting against.
        seekEpochs.isEnabled = false

        Trainer.startTraining(lifecycleScope, lib, epochs = currentEpochs()) { s -> onTrainStatus(s) }
    }

    private fun onTrainStatus(s: TrainStatus) {
        if (s.training) {
            val pct = if (s.totalEpochs > 0) (100 * s.epoch / s.totalEpochs) else 0
            trainProgress.progress = pct
            txtTrainProgress.text = s.message
        } else {
            trainProgress.visibility = android.view.View.GONE
            txtTrainProgress.visibility = android.view.View.GONE
            findViewById<Button>(R.id.btnTrain).isEnabled = true
            seekEpochs.isEnabled = true
            status(s.message)
            melModel = Trainer.loadModel(lib)
            updateModelInfo()
        }
    }

    private fun updateModelInfo() {
        val mm = melModel
        val meta = lib.trainMeta()
        txtModel.text = if (mm == null) {
            "Not trained yet. Add categories + sounds, then hit TRAIN AI. (${lib.totalSamples()} samples ready)"
        } else {
            val loss = meta?.optDouble("final_loss")
            "Trained: ${mm.categories.size} categories (GRU, ~${paramCountKb(mm)} KB)" +
                (if (loss != null && !loss.isNaN()) " • loss ${"%.4f".format(loss)}" else "") +
                " • ${lib.totalSamples()} samples, ${lib.storageBytes() / 1024} KB"
        }
    }

    private fun paramCountKb(m: MelGru): Int {
        val inDim = m.nMels + m.embedDim
        val h = m.hidden
        val count = h * inDim * 3 + h * h * 3 + h * 3 +
            m.nMels * h + m.nMels +
            m.categories.size * m.embedDim
        return (count * 8) / 1024
    }

    // ---- generation ---------------------------------------------------

    private fun generate() {
        val prompt = inputPrompt.text.toString().trim()
        if (prompt.isEmpty()) { toast("Type a prompt"); return }
        val m = matcher
        if (m == null || !m.isTrained) { toast("Train AI first"); return }
        val mm = melModel
        if (mm == null) { toast("Train AI first — no generative model yet"); return }

        val matches = m.match(prompt)
        if (matches.isEmpty()) {
            txtResult.text = "No category matched \"$prompt\".\nTry adding an alias to a category."
            return
        }

        val variation = seekVariation.progress / 100f
        // Read the controls on the main thread before handing off to Default.
        val seconds = currentSeconds()

        // Layers render one at a time, so peak cost is a single layer — but bail
        // out before allocating rather than dying on a long custom duration.
        val needMb = estimateMb(seconds)
        val budgetMb = Runtime.getRuntime().maxMemory() / (1024f * 1024f) * 0.6f
        if (needMb > budgetMb) {
            txtResult.text = String.format(
                Locale.US,
                "%.2fs needs ~%.0f MB per layer, but only ~%.0f MB of heap is safe to use here.\n" +
                    "Shorten the duration.",
                seconds, needMb, budgetMb
            )
            return
        }

        txtResult.text = String.format(
            Locale.US,
            "Generating %.2fs x %d layer(s)...", seconds, matches.size
        )
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) {
                try {
                    Generator.generate(mm, matches, variation, secondsPerLayer = seconds)
                } catch (e: OutOfMemoryError) {
                    GenResult(
                        FloatArray(0),
                        String.format(
                            Locale.US,
                            "Ran out of memory rendering %.2fs. Try a shorter duration.", seconds
                        )
                    )
                }
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
