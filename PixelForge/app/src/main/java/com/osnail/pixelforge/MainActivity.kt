package com.osnail.pixelforge

import android.app.AlertDialog
import android.graphics.Bitmap
import android.graphics.Color
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
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var lib: Library
    private var matcher: NgramModel? = null
    private var pixelModel: PixelGru? = null
    private var lastGenerated: PixelImage? = null

    private lateinit var drawCanvas: PixelCanvasView
    private lateinit var resultCanvas: PixelCanvasView
    private lateinit var paletteStrip: LinearLayout
    private lateinit var inputEpochs: EditText
    private lateinit var inputPrompt: EditText
    private lateinit var seekVariation: SeekBar
    private lateinit var labelVariation: TextView
    private lateinit var txtModel: TextView
    private lateinit var txtResult: TextView
    private lateinit var txtStatus: TextView
    private lateinit var listCategories: LinearLayout
    private lateinit var trainProgress: android.widget.ProgressBar
    private lateinit var txtTrainProgress: TextView

    // ---- file pickers -------------------------------------------------

    private val exportPack = registerForActivityResult(
        ActivityResultContracts.CreateDocument("application/zip")
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        runIo { contentResolver.openOutputStream(uri)!!.use { DataPack.export(lib, it) } }
    }

    private val importPack = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        askMergeMode(uri)
    }

    private val savePng = registerForActivityResult(
        ActivityResultContracts.CreateDocument("image/png")
    ) { uri ->
        val img = lastGenerated
        if (uri == null || img == null) return@registerForActivityResult
        runIo {
            contentResolver.openOutputStream(uri)!!.use { out ->
                val bmp = img.toBitmap()
                val scaled = Bitmap.createScaledBitmap(bmp, GRID_SIZE * 16, GRID_SIZE * 16, false)
                scaled.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            "Saved PNG"
        }
    }

    // ---- lifecycle ------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        drawCanvas = findViewById(R.id.drawCanvas)
        resultCanvas = findViewById(R.id.resultCanvas)
        paletteStrip = findViewById(R.id.paletteStrip)
        inputEpochs = findViewById(R.id.inputEpochs)
        inputPrompt = findViewById(R.id.inputPrompt)
        seekVariation = findViewById(R.id.seekVariation)
        labelVariation = findViewById(R.id.labelVariation)
        txtModel = findViewById(R.id.txtModel)
        txtResult = findViewById(R.id.txtResult)
        txtStatus = findViewById(R.id.txtStatus)
        listCategories = findViewById(R.id.listCategories)
        trainProgress = findViewById(R.id.trainProgress)
        txtTrainProgress = findViewById(R.id.txtTrainProgress)

        lib = Library(this)
        lib.load()
        loadModel()

        drawCanvas.editable = true
        drawCanvas.currentColorIndex = 1
        resultCanvas.editable = false
        buildPaletteStrip()

        seekVariation.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) { labelVariation.text = "Variation: $p%" }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        findViewById<Button>(R.id.btnClearCanvas).setOnClickListener {
            drawCanvas.image = PixelImage()
        }
        findViewById<Button>(R.id.btnSaveToCategory).setOnClickListener { saveToCategoryDialog() }
        findViewById<Button>(R.id.btnTrain).setOnClickListener { train() }
        findViewById<Button>(R.id.btnAddCategory).setOnClickListener { addCategoryDialog() }
        findViewById<Button>(R.id.btnGenerate).setOnClickListener { generate() }
        findViewById<Button>(R.id.btnSavePng).setOnClickListener {
            if (lastGenerated == null) toast("Generate something first")
            else savePng.launch("pixelforge_${System.currentTimeMillis()}.png")
        }
        findViewById<Button>(R.id.btnExport).setOnClickListener {
            exportPack.launch("pixelforge_${System.currentTimeMillis()}.pxpack.zip")
        }
        findViewById<Button>(R.id.btnImport).setOnClickListener {
            importPack.launch(arrayOf("application/zip", "application/octet-stream", "*/*"))
        }

        renderCategories()
        updateModelInfo()
    }

    // ---- palette --------------------------------------------------------

    private fun buildPaletteStrip() {
        paletteStrip.removeAllViews()
        val size = (40 * resources.displayMetrics.density).toInt()
        val margin = (4 * resources.displayMetrics.density).toInt()
        for (i in 0 until Palette.SIZE) {
            val swatch = android.view.View(this)
            val lp = LinearLayout.LayoutParams(size, size)
            lp.setMargins(margin, margin, margin, margin)
            swatch.layoutParams = lp
            swatch.setBackgroundColor(if (i == Palette.TRANSPARENT_IDX) Color.parseColor("#2A2E36") else Palette.colors[i])
            swatch.tag = i
            swatch.setOnClickListener {
                drawCanvas.currentColorIndex = i
                highlightSelectedSwatch(i)
            }
            paletteStrip.addView(swatch)
        }
        highlightSelectedSwatch(drawCanvas.currentColorIndex)
    }

    private fun highlightSelectedSwatch(idx: Int) {
        for (i in 0 until paletteStrip.childCount) {
            val v = paletteStrip.getChildAt(i)
            v.alpha = if ((v.tag as? Int) == idx) 1.0f else 0.5f
        }
    }

    // ---- model ------------------------------------------------------------

    private fun loadModel() {
        matcher = try {
            if (lib.matcherFile.exists()) NgramModel.fromJson(JSONObject(lib.matcherFile.readText())) else null
        } catch (e: Exception) { null }
        pixelModel = Trainer.loadModel(lib)
    }

    private fun train() {
        if (lib.categories.isEmpty()) { toast("Add a category first"); return }
        if (Trainer.isTraining()) { toast("Training already running"); return }
        val epochs = inputEpochs.text.toString().toIntOrNull()
        if (epochs == null || epochs < 10) { toast("Enter an epoch count (at least 10)"); return }

        val m = NgramModel(3)
        m.train(lib.categories)
        matcher = m
        try {
            lib.matcherFile.writeText(m.toJson().toString())
        } catch (e: Exception) {
            // The matcher is live in memory either way; only persistence failed.
            status("Matcher save failed: ${e.message}")
        }

        trainProgress.visibility = android.view.View.VISIBLE
        txtTrainProgress.visibility = android.view.View.VISIBLE
        findViewById<Button>(R.id.btnTrain).isEnabled = false

        Trainer.startTraining(lifecycleScope, lib, epochs = epochs) { s -> onTrainStatus(s) }
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
            status(s.message)
            pixelModel = Trainer.loadModel(lib)
            updateModelInfo()
        }
    }

    private fun updateModelInfo() {
        val pm = pixelModel
        val meta = lib.trainMeta()
        txtModel.text = if (pm == null) {
            "Not trained yet. Draw examples into categories, then hit TRAIN AI. (${lib.totalSamples()} drawings ready)"
        } else {
            val loss = meta?.optDouble("final_loss")
            val epochs = meta?.optInt("epochs")
            "Trained: ${pm.categories.size} categories (GRU, ~${paramCountKb(pm)} KB)" +
                (if (epochs != null && epochs > 0) " • $epochs epochs" else "") +
                (if (loss != null && !loss.isNaN()) " • loss ${"%.4f".format(loss)}" else "") +
                " • ${lib.totalSamples()} drawings, ${lib.storageBytes()} bytes"
        }
    }

    private fun paramCountKb(m: PixelGru): Int {
        val inDim = m.colorEmbedDim * 2 + m.gridSize * 2 + m.embedDim
        val h = m.hidden
        val count = h * inDim * 3 + h * h * 3 + h * 3 +
            m.paletteSize * h + m.paletteSize +
            m.paletteSize * m.colorEmbedDim + m.categories.size * m.embedDim
        return (count * 8) / 1024
    }

    // ---- generation ---------------------------------------------------

    private fun generate() {
        val prompt = inputPrompt.text.toString().trim()
        if (prompt.isEmpty()) { toast("Type a prompt"); return }
        val m = matcher
        if (m == null || !m.isTrained) { toast("Train AI first"); return }
        val pm = pixelModel
        if (pm == null) { toast("Train AI first — no generative model yet"); return }

        val matches = m.match(prompt)
        if (matches.isEmpty()) {
            txtResult.text = "No category matched \"$prompt\".\nTry adding an alias to a category."
            return
        }

        val variation = seekVariation.progress / 100f
        txtResult.text = "Generating..."
        lifecycleScope.launch {
            val result = withContext(Dispatchers.Default) { Generator.generate(pm, matches, variation) }
            if (result.image == null) { txtResult.text = result.log; return@launch }
            lastGenerated = result.image
            resultCanvas.image = result.image
            txtResult.text = result.log
        }
    }

    // ---- categories -----------------------------------------------------

    private fun saveToCategoryDialog() {
        if (drawCanvas.image.isBlank()) { toast("Canvas is empty"); return }
        val names = lib.categories.map { it.name }
        val options = (listOf("+ New category...") + names).toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Save drawing to category")
            .setItems(options) { _, which ->
                if (which == 0) newCategoryThenSave()
                else lib.find(names[which - 1])?.let { saveDrawingTo(it) }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun newCategoryThenSave() {
        val field = EditText(this).apply { hint = "category name (e.g. fish)" }
        AlertDialog.Builder(this)
            .setTitle("New category")
            .setView(field)
            .setPositiveButton("Create & Save") { _, _ ->
                val cat = lib.addCategory(field.text.toString())
                if (cat == null) { toast("Invalid or duplicate name"); return@setPositiveButton }
                saveDrawingTo(cat)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun saveDrawingTo(cat: Category) {
        lib.addSample(cat, drawCanvas.image)
        drawCanvas.image = PixelImage()
        renderCategories()
        updateModelInfo()
        status("Saved to \"${cat.name}\" — TRAIN AI again to include it")
    }

    private fun addCategoryDialog() {
        val name = EditText(this).apply {
            hint = "category name (e.g. boom)"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val aliases = EditText(this).apply {
            hint = "aliases, comma separated"
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 0)
            addView(name); addView(aliases)
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
                status("Category \"${cat.name}\" added — draw something into it, then TRAIN")
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
                text = "No categories yet. Draw something above, then \"Save to Category\"."
                textSize = 12f
                setTextColor(Color.parseColor("#8B98A5"))
            }
            listCategories.addView(tv)
            return
        }
        for (cat in lib.categories) {
            val row = layoutInflater.inflate(R.layout.item_category, listCategories, false)
            row.findViewById<TextView>(R.id.txtCatName).text = cat.name
            row.findViewById<TextView>(R.id.txtCatInfo).text =
                "${cat.samples.size} drawing(s)" + if (cat.aliases.isEmpty()) "" else " • ${cat.aliases.joinToString(", ")}"

            val preview = row.findViewById<PixelCanvasView>(R.id.catPreview)
            preview.editable = false
            if (cat.samples.isNotEmpty()) preview.image = lib.loadSample(cat.samples.random())

            row.findViewById<Button>(R.id.btnAliases).setOnClickListener { aliasDialog(cat) }
            row.findViewById<Button>(R.id.btnPreview).setOnClickListener {
                if (cat.samples.isEmpty()) { toast("No drawings in this category"); return@setOnClickListener }
                preview.image = lib.loadSample(cat.samples.random())
            }
            row.findViewById<Button>(R.id.btnDelete).setOnClickListener {
                AlertDialog.Builder(this)
                    .setTitle("Delete \"${cat.name}\"?")
                    .setMessage("This removes the category and its ${cat.samples.size} drawing(s).")
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

    // ---- data pack ------------------------------------------------------

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
                } catch (e: Exception) { "Import failed: ${e.message}" }
            }
            lib.load()
            loadModel()
            renderCategories()
            updateModelInfo()
            status("$msg — TRAIN again if matching feels off")
        }
    }

    // ---- helpers ----------------------------------------------------------

    private fun runIo(block: suspend () -> Any?) {
        lifecycleScope.launch {
            val msg = withContext(Dispatchers.IO) {
                try { block()?.toString() ?: "Done" } catch (e: Exception) { "Failed: ${e.message}" }
            }
            status(msg)
        }
    }

    private fun status(msg: String) { txtStatus.text = msg }
    private fun toast(msg: String) { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
}
