package com.osnail.pixelforge

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

data class TrainStatus(
    val training: Boolean = false,
    val epoch: Int = 0,
    val totalEpochs: Int = 0,
    val loss: Double? = null,
    val message: String = "Not trained yet.",
    val error: String? = null
)

object Trainer {

    private const val MODEL_FILE = "pixelgru.bin"
    @Volatile private var training = false

    fun isTraining() = training

    fun modelFile(lib: Library): File = File(lib.modelDir, MODEL_FILE)

    fun loadModel(lib: Library): PixelGru? {
        val f = modelFile(lib)
        if (!f.exists()) return null
        return try {
            DataInputStream(FileInputStream(f)).use { PixelGru.readFrom(it) }
        } catch (e: Exception) {
            null
        }
    }

    fun saveModel(lib: Library, model: PixelGru) {
        DataOutputStream(FileOutputStream(modelFile(lib))).use { model.writeTo(it) }
    }

    fun startTraining(
        scope: CoroutineScope,
        lib: Library,
        epochs: Int,
        lr: Double = 0.01,
        onProgress: (TrainStatus) -> Unit
    ) {
        if (training) {
            onProgress(TrainStatus(training = true, message = "Training already running."))
            return
        }
        val cats = lib.categories.filter { it.samples.isNotEmpty() }
        if (cats.isEmpty()) {
            onProgress(TrainStatus(training = false, message = "No categories have any drawings yet."))
            return
        }
        val safeEpochs = epochs.coerceIn(10, 3000)

        training = true
        scope.launch(Dispatchers.Default) {
            try {
                runTraining(lib, cats, safeEpochs, lr, onProgress)
            } catch (e: Exception) {
                withMain(onProgress, TrainStatus(training = false, message = "Training failed: ${e.message}", error = e.message))
            } finally {
                training = false
            }
        }
    }

    private suspend fun withMain(onProgress: (TrainStatus) -> Unit, status: TrainStatus) {
        kotlinx.coroutines.withContext(Dispatchers.Main) { onProgress(status) }
    }

    private suspend fun runTraining(
        lib: Library,
        cats: List<Category>,
        epochs: Int,
        lr: Double,
        onProgress: (TrainStatus) -> Unit
    ) {
        withMain(onProgress, TrainStatus(training = true, totalEpochs = epochs, message = "Loading drawings..."))

        val catNames = cats.map { it.name }
        var model = loadModel(lib)?.takeIf { it.categories == catNames }
        if (model == null) {
            model = PixelGru(
                paletteSize = Palette.SIZE, hidden = 96, colorEmbedDim = 8, embedDim = 16,
                gridSize = GRID_SIZE, categories = catNames.toMutableList()
            )
        }

        data class Item(val catIdx: Int, val pixels: IntArray)
        val dataset = ArrayList<Item>()
        for ((idx, cat) in cats.withIndex()) {
            for (s in cat.samples) {
                val img = lib.loadSample(s)
                if (!img.isBlank()) dataset.add(Item(idx, img.data))
            }
        }

        if (dataset.isEmpty()) {
            withMain(onProgress, TrainStatus(training = false, message = "No usable drawings found.", error = "empty_dataset"))
            return
        }

        withMain(onProgress, TrainStatus(training = true, totalEpochs = epochs,
            message = "Training on ${dataset.size} drawing(s) across ${cats.size} categories..."))

        val order = dataset.indices.toMutableList()
        var lastAvg = 0.0
        // update the UI roughly ~40 times over the whole run, not every single epoch,
        // so a large epoch count doesn't flood the main thread with redraws
        val reportEvery = maxOf(1, epochs / 40)

        for (epoch in 1..epochs) {
            order.shuffle()
            var total = 0.0
            for (i in order) {
                val item = dataset[i]
                total += model.trainSequence(item.pixels, item.catIdx, lr = lr)
            }
            lastAvg = total / order.size

            if (epoch % reportEvery == 0 || epoch == epochs) {
                withMain(onProgress, TrainStatus(
                    training = true, epoch = epoch, totalEpochs = epochs, loss = lastAvg,
                    message = "Epoch $epoch/$epochs — loss ${"%.4f".format(lastAvg)}"
                ))
            }
        }

        saveModel(lib, model)
        lib.saveTrainMeta(catNames, dataset.size, epochs, lastAvg)

        withMain(onProgress, TrainStatus(
            training = false,
            message = "Trained on ${cats.size} categories, ${dataset.size} drawings, $epochs epochs. Final loss ${"%.4f".format(lastAvg)}."
        ))
    }
}
