package com.example.mahjongcalculator

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloDetector(context: Context, modelPath: String) {

    private var interpreter: Interpreter
    private val inputImageWidth = 640
    private val inputImageHeight = 640
    private val confThreshold = 0.20f 
    private val iouThreshold = 0.45f
    
    private val classNames = listOf(
        "1m", "1p", "1s", "1z", 
        "2m", "2p", "2s", "2z",
        "3m", "3p", "3s", "3z", 
        "4m", "4p", "4s", "4z",
        "5m", "5p", "5s", "5z", 
        "6m", "6p", "6s", "6z",
        "7m", "7p", "7s", "7z", 
        "8m", "8p", "8s",
        "9m", "9p", "9s", 
        "UNKNOWN",
        "0m", "0p", "0s"
    )

    init {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        interpreter = Interpreter(model, options)
    }

    data class Detection(
        val classId: Int,
        val className: String,
        val confidence: Float,
        val box: RectF,
        val isRotated: Boolean // New field
    )

    fun detect(bitmap: Bitmap): List<Detection> {
        val letterboxedBitmap = letterbox(bitmap, inputImageWidth, inputImageHeight)
        
        val imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(letterboxedBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape() 
        val channels = outputShape[1] 
        val anchors = outputShape[2] 

        val outputBuffer = ByteBuffer.allocateDirect(1 * channels * anchors * 4)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        interpreter.run(tensorImage.buffer, outputBuffer)
        
        outputBuffer.rewind()
        val floatOutput = FloatArray(channels * anchors)
        outputBuffer.asFloatBuffer().get(floatOutput)

        val detections = ArrayList<Detection>()

        val scale = minOf(
            inputImageWidth.toFloat() / bitmap.width,
            inputImageHeight.toFloat() / bitmap.height
        )
        val padX = (inputImageWidth - bitmap.width * scale) / 2
        val padY = (inputImageHeight - bitmap.height * scale) / 2

        for (i in 0 until anchors) {
            var maxScore = 0f
            var maxClassId = -1
            
            val cx = floatOutput[0 * anchors + i]
            val cy = floatOutput[1 * anchors + i]
            val w = floatOutput[2 * anchors + i]
            val h = floatOutput[3 * anchors + i]
            
            for (c in 0 until (channels - 4)) {
                val score = floatOutput[(4 + c) * anchors + i]
                if (score > maxScore) {
                    maxScore = score
                    maxClassId = c
                }
            }

            if (maxScore > confThreshold) {
                val xInLetterbox = cx - w / 2
                val yInLetterbox = cy - h / 2
                
                val originalX = (xInLetterbox - padX) / scale
                val originalY = (yInLetterbox - padY) / scale
                val originalW = w / scale
                val originalH = h / scale
                
                val rect = RectF(
                    originalX,
                    originalY,
                    originalX + originalW,
                    originalY + originalH
                )
                
                // Determine rotation: if width > height * 1.2 (buffer), it's sideways
                val isRotated = originalW > originalH * 1.1f
                
                val className = if (maxClassId < classNames.size) classNames[maxClassId] else "Unknown"
                detections.add(Detection(maxClassId, className, maxScore, rect, isRotated))
            }
        }

        return nms(detections)
    }

    fun getDebugLetterboxImage(bitmap: Bitmap): Bitmap {
        return letterbox(bitmap, inputImageWidth, inputImageHeight)
    }

    private fun letterbox(source: Bitmap, targetW: Int, targetH: Int): Bitmap {
        val background = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(background)
        // Use Black background for better contrast with Mahjong tiles
        canvas.drawColor(Color.BLACK)

        val scale = minOf(targetW.toFloat() / source.width, targetH.toFloat() / source.height)
        val newW = (source.width * scale).toInt()
        val newH = (source.height * scale).toInt()
        
        val left = (targetW - newW) / 2f
        val top = (targetH - newH) / 2f
        
        // Use high quality filtering
        val paint = Paint().apply { 
            isFilterBitmap = true 
            isAntiAlias = true
            isDither = true
        }
        
        val scaledSrc = Bitmap.createScaledBitmap(source, newW, newH, true)
        canvas.drawBitmap(scaledSrc, left, top, paint)
        
        return background
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = ArrayList<Detection>()
        val active = BooleanArray(sorted.size) { true }
        
        for (i in sorted.indices) {
            if (!active[i]) continue
            val a = sorted[i]
            selected.add(a)
            
            for (j in i + 1 until sorted.size) {
                if (!active[j]) continue
                val b = sorted[j]
                val iou = calculateIoU(a.box, b.box)
                if (iou > iouThreshold) {
                    active[j] = false
                }
            }
        }
        return selected
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val intersectionLeft = maxOf(a.left, b.left)
        val intersectionTop = maxOf(a.top, b.top)
        val intersectionRight = minOf(a.right, b.right)
        val intersectionBottom = minOf(a.bottom, b.bottom)
        
        if (intersectionRight < intersectionLeft || intersectionBottom < intersectionTop) return 0f
        
        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        
        return intersectionArea / (areaA + areaB - intersectionArea)
    }
}
