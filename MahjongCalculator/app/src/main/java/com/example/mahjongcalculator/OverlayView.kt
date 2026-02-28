package com.example.mahjongcalculator

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<DetectionResult> = emptyList()
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        style = Paint.Style.FILL
        isFakeBoldText = true
    }
    private val textBgPaint = Paint().apply {
        color = Color.parseColor("#80000000") // Semi-transparent black
        style = Paint.Style.FILL
    }

    data class DetectionResult(val box: RectF, val text: String)

    fun setResults(detectionResults: List<DetectionResult>) {
        results = detectionResults
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (result in results) {
            canvas.drawRect(result.box, boxPaint)
            
            val textWidth = textPaint.measureText(result.text)
            val textHeight = textPaint.textSize
            
            // Draw text background
            val bgRect = RectF(
                result.box.left,
                result.box.top - textHeight - 10,
                result.box.left + textWidth + 20,
                result.box.top
            )
            
            // If box is too close to top, draw text inside
            var textY = result.box.top - 10
            if (result.box.top < textHeight + 20) {
                bgRect.top = result.box.top
                bgRect.bottom = result.box.top + textHeight + 20
                textY = result.box.top + textHeight + 5
            }
            
            canvas.drawRect(bgRect, textBgPaint)
            canvas.drawText(result.text, result.box.left + 10, textY, textPaint)
        }
    }
}
