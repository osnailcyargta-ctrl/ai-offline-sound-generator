package com.osnail.soundforge

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

/**
 * Runs the whole GRU training loop on a background coroutine so the UI thread
 * never blocks. Progress is reported through a callback that MainActivity
 * hops back to the main thread with, so the progress bar updates live.
 */
object Trainer {

    private const val MODEL_FILE = "melgru.bin"
    @Volatile private var training = false

    fun isTraining() = training

    fun modelFile(lib: Library): File = File(lib.modelDir, MODEL_FILE)

    fun loadModel(lib: Library): MelGru? {
        val f = modelFile(lib)
        if (!f.exists()) return null
        return try {
            DataInputStream(FileInputStream(f)).use { MelGru.readFrom(it) }
        } catch (e: Exception) {
            null
        }
    }

    fun saveModel(lib: Library, model: MelGru) {
        DataOutputStream(FileOutputStream(modelFile(lib))).use { model.writeTo(it) }
    }

    fun startTraining(
        scope: CoroutineScope,
        lib: Library,
        epochs: Int = 150,
        lr: Double = 0.004,
        onProgress: (TrainStatus) -> Unit
    ) {
        if (training) {
            onProgress(TrainStatus(training = true, message = "Training already running."))
            return
        }
        val cats = lib.categories.filter { it.samples.isNotEmpty() }
        if (cats.isEmpty()) {
            onProgress(TrainStatus(training = false, message = "No categories have any sounds yet."))
            return
        }

        training = true
        scope.launch(Dispatchers.Default) {
            try {
                runTraining(lib, cats, epochs, lr, onProgress)
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
        withMain(onProgress, TrainStatus(training = true, totalEpochs = epochs,
            message = "Loading audio & extracting mel features..."))

        val catNames = cats.map { it.name }

        // Reuse existing weights if the category set is unchanged, instead of
        // discarding everything learned so far on every retrain.
        var model = loadModel(lib)?.takeIf { it.categories == catNames }
        if (model == null) {
            model = MelGru(nMels = MelFeatures.N_MELS, hidden = 128, embedDim = 16,
                categories = catNames.toMutableList())
        }

        data class Seq(val catIdx: Int, val mel: Array<FloatArray>)
        val dataset = ArrayList<Seq>()
        for ((idx, cat) in cats.withIndex()) {
            for (s in cat.samples) {
                val audio = lib.loadSample(s)
                val trimmed = Dsp.trimSilence(audio)
                if (trimmed.size < MelFeatures.HOP * 4) continue
                val mel = MelFeatures.audioToMel(trimmed, WavIO.SAMPLE_RATE)
                if (mel.size >= 2) dataset.add(Seq(idx, mel))
            }
        }

        if (dataset.isEmpty()) {
            withMain(onProgress, TrainStatus(training = false,
                message = "No usable audio found (files too short or unreadable).", error = "empty_dataset"))
            return
        }

        withMain(onProgress, TrainStatus(training = true, totalEpochs = epochs,
            message = "Training on ${dataset.size} sound(s) across ${cats.size} categories..."))

        val order = dataset.indices.toMutableList()
        var lastAvg = 0.0
        for (epoch in 1..epochs) {
            order.shuffle()
            var total = 0.0
            for (i in order) {
                val seq = dataset[i]
                total += model.trainSequence(seq.mel, seq.catIdx, lr = lr)
            }
            lastAvg = total / order.size

            withMain(onProgress, TrainStatus(
                training = true, epoch = epoch, totalEpochs = epochs, loss = lastAvg,
                message = "Epoch $epoch/$epochs — loss ${"%.4f".format(lastAvg)}"
            ))
        }

        saveModel(lib, model)
        lib.saveTrainMeta(catNames, dataset.size, lastAvg)

        withMain(onProgress, TrainStatus(
            training = false,
            message = "Trained on ${cats.size} categories, ${dataset.size} sounds. Final loss ${"%.4f".format(lastAvg)}."
        ))
    }
}
