package com.example.mahjongcalculator

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.OrientationEventListener
import android.view.Surface
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TableLayout
import android.widget.TableRow
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.mahjongcalculator.databinding.ActivityMainBinding
import com.yalantis.ucrop.UCrop
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var yoloDetector: YoloDetector? = null
    private val calculator = MahjongCalculator()
    private var currentDetectedTiles: MutableList<String> = mutableListOf()
    private var currentDetections: List<YoloDetector.Detection> = emptyList()
    
    // Selection State
    // Dora Indicators
    private var doraIndicators: MutableList<String> = mutableListOf()
    private var uraDoraIndicators: MutableList<String> = mutableListOf()
    private var selectedWinningTile: String? = null
    
    // Game Context State
    private var prevalentWind = "1z" // Default East
    private var seatWind = "1z"      // Default East
    private var isRiichi = false
    private var isTsumo = false
    private var isIppatsu = false
    
    // Advanced Conditions State
    private var isDoubleRiichi = false
    private var isRinshan = false
    private var isHaitei = false // Haitei/Houtei
    private var isChankan = false
    private var isTenhou = false
    private var isChiihou = false
    
    // UI References for selection rows
    private val doraIndicatorViews = mutableListOf<ImageView>()
    private val uraDoraIndicatorViews = mutableListOf<ImageView>()
    private val winningViews = mutableListOf<ImageView>()

    // Orientation handling
    private var orientationEventListener: OrientationEventListener? = null
    private var deviceOrientation = Surface.ROTATION_0

    private val activityResultLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value) permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext, "Permission request denied", Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }

    // Gallery Picker
    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { startCrop(it) }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: android.content.Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == UCrop.REQUEST_CROP) {
            val resultUri = UCrop.getOutput(data!!)
            resultUri?.let { handleCropResult(it) }
        } else if (resultCode == UCrop.RESULT_ERROR) {
            val cropError = UCrop.getError(data!!)
            Toast.makeText(this, "Crop error: ${cropError?.message}", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        showUsageDialog()

        try {
            yoloDetector = YoloDetector(this, "mahjong_yolo.tflite")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            Toast.makeText(this, "Model not found in assets!", Toast.LENGTH_LONG).show()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        
        setupOrientationListener()
        setupUIListeners()
    }
    
    private fun setupUIListeners() {
        binding.btnLaunchCamera.setOnClickListener { 
            if (allPermissionsGranted()) {
                showCameraUI()
                startCamera()
            } else {
                requestPermissions()
            }
        }

        binding.btnCapture.setOnClickListener { takePhoto() }
        binding.btnCancelCamera.setOnClickListener { showMainUI() }
        
        // Gallery Button
        binding.btnGallery.setOnClickListener { 
            galleryLauncher.launch("image/*")
        }
        
        binding.btnCalculate.setOnClickListener { calculatePoints() }
        
        // Wind Selection
        setupWindSelection()
        
        // Basic Conditions
        binding.chkRiichi.setOnCheckedChangeListener { _, isChecked -> isRiichi = isChecked }
        binding.chkTsumo.setOnCheckedChangeListener { _, isChecked -> isTsumo = isChecked }
        binding.chkIppatsu.setOnCheckedChangeListener { _, isChecked -> isIppatsu = isChecked }
        
        // Complex Conditions
        binding.btnComplexConditions.setOnClickListener { showAdvancedConditionsDialog() }
    }
    
    private fun setupWindSelection() {
        // Prevalent Wind: East(1z) / South(2z)
        binding.imgPrevalentEast.setOnClickListener { 
            prevalentWind = "1z"
            updateWindVisuals(binding.imgPrevalentEast, binding.imgPrevalentSouth, null, null)
        }
        binding.imgPrevalentSouth.setOnClickListener { 
            prevalentWind = "2z"
            updateWindVisuals(binding.imgPrevalentSouth, binding.imgPrevalentEast, null, null)
        }
        
        // Seat Wind: East(1z) / South(2z) / West(3z) / North(4z)
        binding.imgSeatEast.setOnClickListener { 
            seatWind = "1z"
            updateWindVisuals(binding.imgSeatEast, binding.imgSeatSouth, binding.imgSeatWest, binding.imgSeatNorth)
        }
        binding.imgSeatSouth.setOnClickListener { 
            seatWind = "2z"
            updateWindVisuals(binding.imgSeatSouth, binding.imgSeatEast, binding.imgSeatWest, binding.imgSeatNorth)
        }
        binding.imgSeatWest.setOnClickListener { 
            seatWind = "3z"
            updateWindVisuals(binding.imgSeatWest, binding.imgSeatEast, binding.imgSeatSouth, binding.imgSeatNorth)
        }
        binding.imgSeatNorth.setOnClickListener { 
            seatWind = "4z"
            updateWindVisuals(binding.imgSeatNorth, binding.imgSeatEast, binding.imgSeatSouth, binding.imgSeatWest)
        }
    }
    
    private fun updateWindVisuals(selected: ImageView, vararg others: ImageView?) {
        selected.setBackgroundColor(Color.parseColor("#FFD700")) // Gold
        others.forEach { it?.setBackgroundColor(Color.TRANSPARENT) }
    }
    
    private fun showAdvancedConditionsDialog() {
        val options = arrayOf(
            "双立直",
            "岭上开花",
            "海底/河底",
            "枪杠",
            "天和",
            "地和"
        )
        val checkedItems = booleanArrayOf(
            isDoubleRiichi,
            isRinshan,
            isHaitei,
            isChankan,
            isTenhou,
            isChiihou
        )
        
        AlertDialog.Builder(this)
            .setTitle("特殊役种")
            .setMultiChoiceItems(options, checkedItems) { _, which, isChecked ->
                // Logic to prevent conflicts (e.g. Tenhou vs Chiihou)
                if (which == 4 && isChecked) { // Tenhou
                     if (checkedItems[5]) { // If Chiihou checked
                         checkedItems[5] = false
                         (binding.btnComplexConditions.tag as? AlertDialog)?.listView?.setItemChecked(5, false)
                     }
                }
                if (which == 5 && isChecked) { // Chiihou
                     if (checkedItems[4]) { // If Tenhou checked
                         checkedItems[4] = false
                         (binding.btnComplexConditions.tag as? AlertDialog)?.listView?.setItemChecked(4, false)
                     }
                }
                checkedItems[which] = isChecked
            }
            .setPositiveButton("确定") { _, _ ->
                // Save state
                isDoubleRiichi = checkedItems[0]
                isRinshan = checkedItems[1]
                isHaitei = checkedItems[2]
                isChankan = checkedItems[3]
                isTenhou = checkedItems[4]
                isChiihou = checkedItems[5]
                
                // Update Summary Text
                val summary = mutableListOf<String>()
                if (isDoubleRiichi) summary.add("双立直")
                if (isRinshan) summary.add("岭上开花")
                if (isHaitei) summary.add("海底/河底")
                if (isChankan) summary.add("枪杠")
                if (isTenhou) summary.add("天和")
                if (isChiihou) summary.add("地和")
                
                binding.txtAdvancedSummary.text = if (summary.isEmpty()) "未选择" else summary.joinToString(", ")
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun startCrop(uri: Uri) {
        val destinationFileName = "cropped_mahjong_image.jpg"
        val destinationUri = Uri.fromFile(File(cacheDir, destinationFileName))
        
        // UCrop Configuration
        val options = UCrop.Options()
        options.setCompressionQuality(90)
        options.setFreeStyleCropEnabled(true) // Allow any aspect ratio
        options.setToolbarTitle("截取手牌区域")
        options.setToolbarColor(ContextCompat.getColor(this, android.R.color.black))
        options.setStatusBarColor(ContextCompat.getColor(this, android.R.color.black))
        options.setToolbarWidgetColor(ContextCompat.getColor(this, android.R.color.white)) // Set icon/text color to white
        options.setActiveControlsWidgetColor(ContextCompat.getColor(this, android.R.color.holo_blue_light))
        
        UCrop.of(uri, destinationUri)
            .withOptions(options)
            .withAspectRatio(16f, 3f) // Optional default strip aspect
            .start(this)
    }
    
    private fun handleCropResult(uri: Uri) {
        try {
            val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
            
            showMainUI()
            binding.resultImageView.setImageBitmap(bitmap)
            binding.placeholderText.visibility = View.GONE
            
            analyzeImage(bitmap)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error handling crop result", e)
            Toast.makeText(this, "图片处理失败", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showUsageDialog() {
        val message = """
            【使用流程】
            1. 点击 "拍摄手牌"
            2. 将麻将牌置于取景框中央
            3. 系统会自动识别并排序
            4. 在下方两行麻将中分别选择宝牌和胡牌
            5. 点击计算

            【注意事项】
            📷 拍摄：竖屏拍摄，保持光线充足
            🀄 暗杠：请将暗杠的4张牌全部翻开拍摄
            🔁 鸣牌：横向放置的牌会被自动识别并旋转
            🔴 赤牌：红色5会被自动识别并优先处理
        """.trimIndent()

        AlertDialog.Builder(this)
            .setTitle("使用须知")
            .setMessage(message)
            .setPositiveButton("知道了", null)
            .setCancelable(false)
            .show()
    }
    
    private fun sortAndDisplayTiles() {
        if (currentDetections.isEmpty()) return

        // Sort by Spatial Position (X-coordinate)
        val sortedDetections = currentDetections.sortedBy { it.box.left }

        // Update Global State
        currentDetectedTiles = sortedDetections.map { it.className }.toMutableList()
        
        // Reset selections
        doraIndicators.clear()
        uraDoraIndicators.clear()
        selectedWinningTile = null
        binding.txtSelectedWinning.text = "当前选择：无"
        
        // Display all tiles in the main container
        binding.tilesContainer.removeAllViews()
        binding.meldsContainer.removeAllViews() 
        
        // Ankan Visual Logic (Apply to main list)
        val displayTiles = sortedDetections.map { it.className to it.isRotated }.toMutableList()
        if (displayTiles.size >= 4) {
            var i = 0
            while (i <= displayTiles.size - 4) {
                val t1 = displayTiles[i].first
                val t2 = displayTiles[i+1].first
                val t3 = displayTiles[i+2].first
                val t4 = displayTiles[i+3].first
                
                fun norm(t: String) = if(t.startsWith("0")) "5" + t.drop(1) else t
                
                if (norm(t1) == norm(t2) && norm(t2) == norm(t3) && norm(t3) == norm(t4)) {
                    displayTiles[i] = "flat_back" to displayTiles[i].second
                    displayTiles[i+3] = "flat_back" to displayTiles[i+3].second
                    i += 4 
                } else {
                    i++
                }
            }
        }
        
        populateTileContainer(binding.tilesContainer, displayTiles)
        
        // Populate Selection Rows
        
        // Winning Row: Show all tiles, upright
        val winningTiles = currentDetectedTiles.filter { tile ->
            val normTile = if (tile.startsWith("0")) "5" + tile.drop(1) else tile
            val count = currentDetectedTiles.count { 
                val t = if (it.startsWith("0")) "5" + it.drop(1) else it
                t == normTile 
            }
            count != 4 
        }.map { it to false }

        populateSelectionRow(binding.winningSelectionContainer, winningViews, false, winningTiles)
        
        // Dora Indicators: Initialize with just "+" button
        populateDoraIndicatorRow(binding.doraIndicatorContainer, doraIndicatorViews, false)
        
        // Ura Dora Indicators: Initialize with just "+" button
        populateDoraIndicatorRow(binding.uraDoraIndicatorContainer, uraDoraIndicatorViews, true)
        
        binding.txtResult.text = "识别到 ${currentDetectedTiles.size} 张牌。\n请选择胡牌并设置宝牌指示牌。"
    }
    
    private fun populateTileContainer(container: LinearLayout, tiles: List<Pair<String, Boolean>>) {
        for ((index, pair) in tiles.withIndex()) {
            val tile = pair.first
            val isRotated = pair.second
            
            val imageView = ImageView(this)
            var layoutW = 75
            var layoutH = 105
            
            if (isRotated) {
                layoutW = 105
                layoutH = 75
            }
            
            val params = LinearLayout.LayoutParams(layoutW, layoutH)
            params.setMargins(0, 0, 0, 0)
            params.gravity = android.view.Gravity.BOTTOM
            imageView.layoutParams = params
            imageView.scaleType = ImageView.ScaleType.FIT_CENTER
            
            val resId = getTileDrawableId(tile)
            if (resId != 0) {
                if (isRotated) {
                    val originalBitmap = BitmapFactory.decodeResource(resources, resId)
                    val matrix = Matrix()
                    matrix.postRotate(-90f)
                    val rotatedBitmap = Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true)
                    imageView.setImageBitmap(rotatedBitmap)
                } else {
                    imageView.setImageResource(resId)
                }
            } else {
                imageView.setImageResource(android.R.drawable.ic_menu_help)
            }
            
            // Add click listener for manual correction
            imageView.setOnClickListener {
                showTileCorrectionDialog(index, tile)
            }
            
            container.addView(imageView)
        }

        // Add "Add Tile" button at the end
        val addBtn = ImageView(this)
        val params = LinearLayout.LayoutParams(75, 105)
        params.setMargins(8, 0, 8, 0)
        params.gravity = android.view.Gravity.BOTTOM
        addBtn.layoutParams = params
        addBtn.scaleType = ImageView.ScaleType.FIT_CENTER
        addBtn.setImageResource(android.R.drawable.ic_input_add) // Use system add icon or custom
        addBtn.setColorFilter(Color.parseColor("#4CAF50")) // Green color for add button
        
        addBtn.setOnClickListener {
            showAddTileDialog()
        }
        container.addView(addBtn)
    }

    private fun showAddTileDialog() {
        val allTiles = listOf(
            "1m", "2m", "3m", "4m", "5m", "0m", "6m", "7m", "8m", "9m",
            "1p", "2p", "3p", "4p", "5p", "0p", "6p", "7p", "8p", "9p",
            "1s", "2s", "3s", "4s", "5s", "0s", "6s", "7s", "8s", "9s",
            "1z", "2z", "3z", "4z", "5z", "6z", "7z"
        )
        
        val scrollView = android.widget.ScrollView(this)
        val gridLayout = android.widget.GridLayout(this).apply {
            columnCount = 6
            setPadding(20, 20, 20, 20)
        }
        scrollView.addView(gridLayout)
        
        val dialog = AlertDialog.Builder(this)
            .setTitle("添加新牌")
            .setView(scrollView)
            .setNegativeButton("取消", null)
            .create()
            
        allTiles.forEach { tileName ->
            val img = ImageView(this).apply {
                val size = 100
                layoutParams = android.widget.GridLayout.LayoutParams().apply {
                    width = size
                    height = (size * 1.4).toInt()
                    setMargins(8, 8, 8, 8)
                }
                scaleType = ImageView.ScaleType.FIT_CENTER
                setImageResource(getTileDrawableId(tileName))
                setOnClickListener {
                    currentDetectedTiles.add(tileName)
                    refreshDisplayAfterCorrection()
                    dialog.dismiss()
                }
            }
            gridLayout.addView(img)
        }
        
        dialog.show()
    }
    
    private fun showTileCorrectionDialog(index: Int, currentTile: String) {
        val allTiles = listOf(
            "1m", "2m", "3m", "4m", "5m", "0m", "6m", "7m", "8m", "9m",
            "1p", "2p", "3p", "4p", "5p", "0p", "6p", "7p", "8p", "9p",
            "1s", "2s", "3s", "4s", "5s", "0s", "6s", "7s", "8s", "9s",
            "1z", "2z", "3z", "4z", "5z", "6z", "7z"
        )
        
        // Create a custom dialog layout
        val scrollView = android.widget.ScrollView(this)
        val gridLayout = android.widget.GridLayout(this).apply {
            columnCount = 6
            setPadding(20, 20, 20, 20)
        }
        scrollView.addView(gridLayout)
        
        val dialog = AlertDialog.Builder(this)
            .setTitle("修正牌张")
            .setView(scrollView)
            .setNeutralButton("删除此牌") { _, _ ->
                currentDetectedTiles.removeAt(index)
                // Refresh UI
                refreshDisplayAfterCorrection()
            }
            .setNegativeButton("取消", null)
            .create()
            
        // Populate Grid
        allTiles.forEach { tileName ->
            val img = ImageView(this).apply {
                val size = 100
                layoutParams = android.widget.GridLayout.LayoutParams().apply {
                    width = size
                    height = (size * 1.4).toInt()
                    setMargins(8, 8, 8, 8)
                }
                scaleType = ImageView.ScaleType.FIT_CENTER
                setImageResource(getTileDrawableId(tileName))
                setOnClickListener {
                    // Update the tile at this index
                    if (index < currentDetectedTiles.size) {
                        currentDetectedTiles[index] = tileName
                        refreshDisplayAfterCorrection()
                    }
                    dialog.dismiss()
                }
            }
            gridLayout.addView(img)
        }
        
        dialog.show()
    }
    
    private fun refreshDisplayAfterCorrection() {
        // Re-render Main Tiles
        binding.tilesContainer.removeAllViews()
        val displayPairs = currentDetectedTiles.map { it to false } // Reset to upright for simplicity after edit
        populateTileContainer(binding.tilesContainer, displayPairs)
        
        // Re-render Selection Rows
        val winningTiles = currentDetectedTiles.filter { tile ->
            val normTile = if (tile.startsWith("0")) "5" + tile.drop(1) else tile
            val count = currentDetectedTiles.count { 
                val t = if (it.startsWith("0")) "5" + it.drop(1) else it
                t == normTile 
            }
            count != 4 
        }.map { it to false }
        
        populateSelectionRow(binding.winningSelectionContainer, winningViews, false, winningTiles)
        
        val uniqueDoraTiles = currentDetectedTiles.distinct().map { it to false }
        
        binding.txtResult.text = "已手动修正。识别到 ${currentDetectedTiles.size} 张牌。"
    }
    
    private fun populateDoraIndicatorRow(
        container: LinearLayout, 
        viewList: MutableList<ImageView>, 
        isUra: Boolean
    ) {
        container.removeAllViews()
        viewList.clear()
        
        val indicators = if (isUra) uraDoraIndicators else doraIndicators
        
        // Add existing indicators
        indicators.forEachIndexed { index, tile ->
            val imageView = ImageView(this)
            val layoutW = 75
            val layoutH = 105
            val params = LinearLayout.LayoutParams(layoutW, layoutH) 
            params.setMargins(4, 0, 4, 0)
            params.gravity = android.view.Gravity.BOTTOM
            imageView.layoutParams = params
            imageView.scaleType = ImageView.ScaleType.FIT_CENTER
            
            val resId = getTileDrawableId(tile)
            if (resId != 0) imageView.setImageResource(resId)
            
            // Click to remove
            imageView.setOnClickListener {
                if (isUra) uraDoraIndicators.removeAt(index) else doraIndicators.removeAt(index)
                populateDoraIndicatorRow(container, viewList, isUra)
            }
            
            container.addView(imageView)
            viewList.add(imageView)
        }
        
        // Add "+" Button
        val addBtn = ImageView(this)
        val params = LinearLayout.LayoutParams(75, 105)
        params.setMargins(8, 0, 8, 0)
        params.gravity = android.view.Gravity.BOTTOM
        addBtn.layoutParams = params
        addBtn.scaleType = ImageView.ScaleType.FIT_CENTER
        addBtn.setImageResource(android.R.drawable.ic_input_add)
        addBtn.setColorFilter(Color.parseColor("#FFD700")) // Gold
        
        addBtn.setOnClickListener {
            showDoraIndicatorDialog(isUra)
        }
        container.addView(addBtn)
    }

    private fun showDoraIndicatorDialog(isUra: Boolean) {
         val allTiles = listOf(
             "1m", "2m", "3m", "4m", "5m", "0m", "6m", "7m", "8m", "9m",
             "1p", "2p", "3p", "4p", "5p", "0p", "6p", "7p", "8p", "9p",
             "1s", "2s", "3s", "4s", "5s", "0s", "6s", "7s", "8s", "9s",
             "1z", "2z", "3z", "4z", "5z", "6z", "7z"
         )
         
         val scrollView = android.widget.ScrollView(this)
         val gridLayout = android.widget.GridLayout(this).apply {
             columnCount = 6
             setPadding(20, 20, 20, 20)
         }
         scrollView.addView(gridLayout)
         
         val title = if (isUra) "添加里宝牌指示牌" else "添加宝牌指示牌"
         val dialog = AlertDialog.Builder(this)
             .setTitle(title)
             .setView(scrollView)
             .setNegativeButton("取消", null)
             .create()
             
         allTiles.forEach { tileName ->
             val img = ImageView(this).apply {
                 val size = 100
                 layoutParams = android.widget.GridLayout.LayoutParams().apply {
                     width = size
                     height = (size * 1.4).toInt()
                     setMargins(8, 8, 8, 8)
                 }
                 scaleType = ImageView.ScaleType.FIT_CENTER
                 setImageResource(getTileDrawableId(tileName))
                 setOnClickListener {
                     if (isUra) {
                         uraDoraIndicators.add(tileName)
                         populateDoraIndicatorRow(binding.uraDoraIndicatorContainer, uraDoraIndicatorViews, true)
                     } else {
                         doraIndicators.add(tileName)
                         populateDoraIndicatorRow(binding.doraIndicatorContainer, doraIndicatorViews, false)
                     }
                     dialog.dismiss()
                 }
             }
             gridLayout.addView(img)
         }
         
         dialog.show()
    }
    
    // Updated signature to take specific tiles list
    private fun populateSelectionRow(
        container: LinearLayout, 
        viewList: MutableList<ImageView>, 
        isDora: Boolean, 
        tilesToDisplay: List<Pair<String, Boolean>>
    ) {
        container.removeAllViews()
        viewList.clear()
        
        tilesToDisplay.forEachIndexed { index, (tile, isRotated) ->
            val imageView = ImageView(this)
            // Force upright size for selection rows
            val layoutW = 75
            val layoutH = 105

            val params = LinearLayout.LayoutParams(layoutW, layoutH) 
            params.setMargins(4, 0, 4, 0)
            params.gravity = android.view.Gravity.BOTTOM
            imageView.layoutParams = params
            imageView.scaleType = ImageView.ScaleType.FIT_CENTER
            
            val resId = getTileDrawableId(tile)
            if (resId != 0) {
                // Always load normal upright image for selection
                imageView.setImageResource(resId)
            } else {
                imageView.setImageResource(android.R.drawable.ic_menu_help)
            }
            
            imageView.setOnClickListener {
                if (!isDora) {
                    selectedWinningTile = tile
                    binding.txtSelectedWinning.text = "当前选择：$tile"
                    updateWinningSelectionVisuals(winningViews, index)
                }
            }
            
            container.addView(imageView)
            viewList.add(imageView)
        }
    }
    
    private fun updateWinningSelectionVisuals(views: List<ImageView>, selectedIndex: Int) {
        views.forEachIndexed { i, view ->
            if (i == selectedIndex) {
                view.setBackgroundColor(Color.parseColor("#FF0000")) // Red for winning
                view.setPadding(4, 4, 4, 4)
            } else {
                view.setBackgroundColor(Color.TRANSPARENT)
                view.setPadding(0, 0, 0, 0)
            }
        }
    }
    
    private fun setupOrientationListener() {
        orientationEventListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation == OrientationEventListener.ORIENTATION_UNKNOWN) return
                
                val rotation = when (orientation) {
                    in 45..134 -> Surface.ROTATION_270
                    in 135..224 -> Surface.ROTATION_180
                    in 225..314 -> Surface.ROTATION_90
                    else -> Surface.ROTATION_0
                }
                deviceOrientation = rotation
                imageCapture?.targetRotation = rotation
            }
        }
    }
    
    override fun onStart() {
        super.onStart()
        orientationEventListener?.enable()
    }

    override fun onStop() {
        super.onStop()
        orientationEventListener?.disable()
    }

    private fun showMainUI() {
        binding.mainContainer.visibility = View.VISIBLE
        binding.cameraContainer.visibility = View.GONE
    }

    private fun showCameraUI() {
        binding.mainContainer.visibility = View.GONE
        binding.cameraContainer.visibility = View.VISIBLE
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder().build()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return
        imageCapture.targetRotation = deviceOrientation
        
        imageCapture.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bitmap = imageProxyToBitmap(image)
                image.close()
                if (bitmap != null) {
                    // Auto-Crop Logic
                    val cropH = (bitmap.height * 0.25).toInt() 
                    val cropY = (bitmap.height - cropH) / 2
                    val croppedBitmap = Bitmap.createBitmap(bitmap, 0, cropY, bitmap.width, cropH)
                    
                    runOnUiThread { 
                        showMainUI()
                        binding.resultImageView.setImageBitmap(croppedBitmap)
                        binding.placeholderText.visibility = View.GONE
                        analyzeImage(croppedBitmap)
                    }
                }
            }
            override fun onError(exc: ImageCaptureException) {
                Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
            }
        })
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        
        val rotationDegrees = image.imageInfo.rotationDegrees
        if (rotationDegrees != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }
        return bitmap
    }
    

    private fun analyzeImage(bitmap: Bitmap) {
        if (yoloDetector == null) return
        
        cameraExecutor.execute {
            try {
                val detections = yoloDetector!!.detect(bitmap)
                
                runOnUiThread {
                    currentDetections = detections
                    val overlayResults = detections.map { 
                        OverlayView.DetectionResult(it.box, "${it.className} ${(it.confidence * 100).toInt()}%") 
                    }
                    binding.detectionOverlay.setResults(overlayResults)
                    
                    // DEBUG: Show model input image on click
                    binding.detectionOverlay.setOnClickListener {
                        if (yoloDetector != null) {
                            val debugBitmap = yoloDetector!!.getDebugLetterboxImage(bitmap)
                            val debugView = ImageView(this@MainActivity)
                            debugView.setImageBitmap(debugBitmap)
                            debugView.adjustViewBounds = true
                            
                            AlertDialog.Builder(this@MainActivity)
                                .setTitle("模型输入视角 (640x640)")
                                .setView(debugView)
                                .setPositiveButton("关闭", null)
                                .show()
                        }
                    }
                    
                    currentDetectedTiles = detections.map { it.className }.toMutableList()
                    sortAndDisplayTiles()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Detection error", e)
            }
        }
    }
    
    private fun getTileType(tile: String): Int {
        return when {
            tile.endsWith("m") -> 1
            tile.endsWith("p") -> 2
            tile.endsWith("s") -> 3
            tile.endsWith("z") -> 4
            else -> 5
        }
    }
    
    private fun getTileValue(tile: String): Int {
        if (tile.length < 2) return 0
        return tile[0].digitToIntOrNull() ?: 0
    }
    
    private fun getTileDrawableId(tileName: String): Int {
        if (tileName == "flat_back") {
            return resources.getIdentifier("flat_back", "drawable", packageName)
        }
        val resourceName = "tile_${tileName.lowercase()}"
        return resources.getIdentifier(resourceName, "drawable", packageName)
    }

    private fun calculatePoints() {
        if (selectedWinningTile == null) {
            Toast.makeText(this, "请先选择胡牌！", Toast.LENGTH_SHORT).show()
            return
        }
        
        val context = MahjongCalculator.GameContext(
            prevalentWind = prevalentWind, 
            seatWind = seatWind,           
            doraIndicators = doraIndicators, 
            uraDoraIndicators = uraDoraIndicators,
            isRiichi = isRiichi,
            isTsumo = isTsumo,
            isIppatsu = isIppatsu,
            isDoubleRiichi = isDoubleRiichi,
            isRinshan = isRinshan,
            isHaitei = isHaitei,
            isChankan = isChankan,
            isTenhou = isTenhou,
            isChiihou = isChiihou,
            honba = binding.spinnerHonba.selectedItemPosition
        )

        val result = calculator.calculate(currentDetectedTiles, selectedWinningTile!!, context)
        
        // Clear previous results
        val resultContainer = findViewById<LinearLayout>(R.id.resultContainer)
        resultContainer.removeAllViews()

        if (result.error != null) {
            val errorView = TextView(this)
            errorView.text = "错误: ${result.error}"
            errorView.textSize = 18f
            errorView.setTextColor(Color.RED)
            resultContainer.addView(errorView)
        } else {
            // 1. Yaku Table (Two Columns)
            val tableLayout = TableLayout(this)
            tableLayout.layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            tableLayout.isStretchAllColumns = true
            
            result.yakuList.forEach { yaku ->
                val row = TableRow(this)
                val nameView = TextView(this)
                val valueView = TextView(this)
                
                var name = yaku
                var value = ""
                
                if (yaku.contains("(")) {
                    val parts = yaku.split("(")
                    name = parts[0].trim()
                    value = parts[1].replace(")", "").trim()
                }
                
                nameView.text = name
                nameView.textSize = 16f
                nameView.setTypeface(Typeface.DEFAULT_BOLD) // Chinese Bold
                nameView.setTextColor(Color.BLACK)
                
                valueView.text = value
                valueView.textSize = 16f
                valueView.setTypeface(Typeface.MONOSPACE) // Consolas-like
                valueView.setTextColor(Color.BLACK)
                valueView.gravity = android.view.Gravity.END
                
                row.addView(nameView)
                row.addView(valueView)
                row.setPadding(0, 4, 0, 4)
                tableLayout.addView(row)
            }
            resultContainer.addView(tableLayout)
            
            // Divider
            val divider = View(this)
            divider.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 2)
            divider.setBackgroundColor(Color.LTGRAY)
            val divParams = divider.layoutParams as LinearLayout.LayoutParams
            divParams.setMargins(0, 16, 0, 16)
            divider.layoutParams = divParams
            resultContainer.addView(divider)
            
            // 2. Han & Fu Row
            val statsRow = LinearLayout(this)
            statsRow.orientation = LinearLayout.HORIZONTAL
            
            val hanView = TextView(this)
            hanView.text = "番数: ${result.han}番"
            hanView.textSize = 16f
            hanView.typeface = Typeface.MONOSPACE
            hanView.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            
            val fuView = TextView(this)
            fuView.text = "符数: ${result.fu}符"
            fuView.textSize = 16f
            fuView.typeface = Typeface.MONOSPACE
            fuView.gravity = android.view.Gravity.END
            fuView.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            
            statsRow.addView(hanView)
            statsRow.addView(fuView)
            resultContainer.addView(statsRow)
            
            // 3. Points Row
            val scoreStr = if (result.scoreDetails.isNotEmpty()) result.scoreDetails else result.points.toString()
            val pointsView = TextView(this)
            pointsView.text = "点数: $scoreStr"
            pointsView.textSize = 20f
            pointsView.setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            pointsView.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                topMargin = 16
            }
            resultContainer.addView(pointsView)
        }
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private const val TAG = "MahjongApp"
        private val REQUIRED_PERMISSIONS =
            mutableListOf(Manifest.permission.CAMERA).toTypedArray()
    }
}
