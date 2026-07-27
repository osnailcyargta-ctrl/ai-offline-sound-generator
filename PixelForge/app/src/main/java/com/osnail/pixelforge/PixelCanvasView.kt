package com.osnail.pixelforge

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.View.MeasureSpec

class PixelCanvasView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    var image: PixelImage = PixelImage()
        set(value) { field = value; invalidate() }

    var editable: Boolean = true
    var currentColorIndex: Int = 1
    var onPixelPainted: (() -> Unit)? = null

    private val cellPaint = Paint()
    private val gridPaint = Paint().apply {
        color = Color.parseColor("#2B3441")
        strokeWidth = 1f
    }
    private val checkerLight = Paint().apply { color = Color.parseColor("#2A2E36") }
    private val checkerDark = Paint().apply { color = Color.parseColor("#1E2229") }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val widthSize = MeasureSpec.getSize(widthMeasureSpec)
        setMeasuredDimension(widthSize, widthSize)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cell = width.toFloat() / GRID_SIZE
        for (y in 0 until GRID_SIZE) {
            for (x in 0 until GRID_SIZE) {
                val idx = image.get(x, y)
                val left = x * cell; val top = y * cell
                if (idx == Palette.TRANSPARENT_IDX) {
                    val checker = if ((x + y) % 2 == 0) checkerLight else checkerDark
                    canvas.drawRect(left, top, left + cell, top + cell, checker)
                } else {
                    cellPaint.color = Palette.colors[idx]
                    canvas.drawRect(left, top, left + cell, top + cell, cellPaint)
                }
            }
        }
        // grid lines every pixel keep it readable while drawing
        for (i in 0..GRID_SIZE) {
            val p = i * cell
            canvas.drawLine(p, 0f, p, height.toFloat(), gridPaint)
            canvas.drawLine(0f, p, width.toFloat(), p, gridPaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!editable) return false
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                val cell = width.toFloat() / GRID_SIZE
                val x = (event.x / cell).toInt().coerceIn(0, GRID_SIZE - 1)
                val y = (event.y / cell).toInt().coerceIn(0, GRID_SIZE - 1)
                if (image.get(x, y) != currentColorIndex) {
                    image.set(x, y, currentColorIndex)
                    invalidate()
                    onPixelPainted?.invoke()
                }
                return true
            }
        }
        return false
    }
}
