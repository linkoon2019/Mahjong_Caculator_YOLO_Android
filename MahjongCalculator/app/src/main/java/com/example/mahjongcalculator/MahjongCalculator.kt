package com.example.mahjongcalculator

import kotlin.math.ceil
import kotlin.math.pow

class MahjongCalculator {

    data class GameContext(
        val prevalentWind: String, // "1z"=East, "2z"=South
        val seatWind: String,      // "1z"..
        val doraIndicators: List<String> = emptyList(), // Indicators for Dora
        val uraDoraIndicators: List<String> = emptyList(), // Indicators for Ura Dora
        val isRiichi: Boolean = false,
        val isTsumo: Boolean = false,
        val isIppatsu: Boolean = false,
        val isDoubleRiichi: Boolean = false,
        val isRinshan: Boolean = false,
        val isHaitei: Boolean = false, // Haitei (Tsumo) or Houtei (Ron)
        val isChankan: Boolean = false,
        val isTenhou: Boolean = false,
        val isChiihou: Boolean = false,
        val honba: Int = 0 // Number of honba (repeats), e.g. 300 points per honba
    )

    data class ScoreResult(
        val han: Int,
        val fu: Int,
        val points: Int, // Total points (or payment from non-dealer in Ron)
        val yakuList: List<String>,
        val error: String? = null,
        val scoreDetails: String = "" // For display: "4000/2000" etc.
    )

    // Internal Tile Representation
    data class Tile(
        val original: String,
        val suit: Int, // 1=Man, 2=Pin, 3=Sou, 4=Zi
        val value: Int, // 1-9
        val isRed: Boolean,
        val isRotated: Boolean = false // Hint for open melds (not fully used yet)
    ) : Comparable<Tile> {
        override fun compareTo(other: Tile): Int {
            if (suit != other.suit) return suit - other.suit
            return value - other.value
        }
        
        fun isSimple(): Boolean = suit != 4 && value in 2..8
        fun isTerminal(): Boolean = suit != 4 && (value == 1 || value == 9)
        fun isHonor(): Boolean = suit == 4
        fun isTerminalOrHonor(): Boolean = isTerminal() || isHonor()
        
        // Helper to get next tile in sequence (wrapping not allowed, but useful for lookup)
        fun next(): Tile? = if (suit != 4 && value < 9) copy(value = value + 1, original = "") else null
        
        // Equality for checking matches (ignores Red/Rotated status)
        fun isSameType(other: Tile): Boolean = suit == other.suit && value == other.value
    }

    fun calculate(
        tilesRaw: List<String>, 
        winningTileRaw: String, 
        context: GameContext
    ): ScoreResult {
        // 1. Parse and Sort Tiles
        val allTiles = tilesRaw.map { parseTile(it) }.sorted()
        val winningTile = parseTile(winningTileRaw)
        
        // Count Validation
        // Valid: 14, 15(1Kan), 16(2Kan), 17(3Kan), 18(4Kan)
        if (allTiles.size !in 14..18) {
             return ScoreResult(0, 0, 0, emptyList(), "牌数错误: ${allTiles.size}张。标准手牌应为14张（含胡牌）。")
        }

        // 2. Partition Hand (Find all possible arrangements)
        // We treat all input tiles as the "Hand". 
        // Note: Currently we assume CLOSED HAND (Menzen) because UI doesn't specify open melds.
        // However, if we detect 4 identical tiles, we might treat as Kan? 
        // For now, standard backtracking.
        
        val partitions = PartitionHelper.partition(allTiles)
        
        if (partitions.isEmpty()) {
            return ScoreResult(0, 0, 0, emptyList(), "未听牌或无役 (No valid hand structure found)")
        }

        // 3. Evaluate Yaku and Score for each partition, pick the best
        var bestResult: ScoreResult? = null
        var maxPoints = -1

        for (partition in partitions) {
            val result = evaluatePartition(partition, allTiles, winningTile, context)
            if (result.error == null && result.points > maxPoints) {
                maxPoints = result.points
                bestResult = result
            }
        }

        return bestResult ?: ScoreResult(0, 0, 0, emptyList(), "无役 (No Yaku)")
    }

    private fun parseTile(str: String): Tile {
        if (str.length < 2) return Tile(str, 5, 0, false) // Unknown
        val typeChar = str.last()
        val valChar = str.first()
        
        val suit = when (typeChar) {
            'm' -> 1
            'p' -> 2
            's' -> 3
            'z' -> 4
            else -> 5
        }
        
        var value = valChar.digitToIntOrNull() ?: 0
        val isRed = value == 0
        if (isRed) value = 5
        
        return Tile(str, suit, value, isRed)
    }

    // --- Evaluation Logic ---

    private fun evaluatePartition(
        partition: Partition, 
        allTiles: List<Tile>, 
        winningTile: Tile, 
        ctx: GameContext
    ): ScoreResult {
        val yakuList = mutableListOf<String>()
        var han = 0
        
        // 1. Yakuman Checks (Simplified: Kokushi, Suuankou, etc.)
        // If partition is standard, check standard Yaku
        
        // --- Standard Yaku ---
        
        // Riichi / Double Riichi
        if (ctx.isDoubleRiichi) { yakuList.add("双立直 (2番)"); han += 2 }
        else if (ctx.isRiichi) { yakuList.add("立直 (1番)"); han += 1 }
        
        // Ippatsu
        if (ctx.isRiichi && ctx.isIppatsu) { yakuList.add("一发 (1番)"); han += 1 }
        
        // Tsumo (Menzen Chin Tsumo) - Assuming Menzen for now
        if (ctx.isTsumo) { yakuList.add("门前清自摸和 (1番)"); han += 1 }
        
        // Pinfu (Pinghu)
        // Conditions: Menzen, All Shuntsu, Pair is not Value (Dragons/Seat/Prevalent), Wait is ryanmen
        if (isPinfu(partition, ctx, winningTile)) {
            yakuList.add("平和 (1番)"); han += 1
        }
        
        // Tanyao (All Simples)
        if (allTiles.all { it.isSimple() }) {
            yakuList.add("断幺九 (1番)"); han += 1
        }
        
        // Iipeiko (Pure Double Chow) / Ryanpeiko (Twice Pure Double Chow)
        if (partition.type == PartitionType.STANDARD) {
            val identicalCount = countIdenticalShuntsu(partition.sets)
            // countIdenticalShuntsu returns number of matching pairs:
            // 2 sets same -> 1 match
            // 3 sets same -> 3 matches
            // 4 sets same -> 6 matches
            // 2 sets same + 2 sets same -> 1 + 1 = 2 matches
            
            if (identicalCount == 2 || identicalCount == 6) {
                 yakuList.add("二杯口 (3番)"); han += 3
            } else if (identicalCount == 1 || identicalCount == 3) {
                 yakuList.add("一杯口 (1番)"); han += 1
            }
        }
        
        // Yakuhai (Value Tiles)
        // Dragons
        partition.sets.filter { it.type == SetType.KOUTSU || it.type == SetType.KANTSU }.forEach { set ->
            val t = set.tiles[0]
            if (t.suit == 4) {
                if (t.value == 5) { yakuList.add("白 (1番)"); han += 1 }
                if (t.value == 6) { yakuList.add("发 (1番)"); han += 1 }
                if (t.value == 7) { yakuList.add("中 (1番)"); han += 1 }
            }
            // Winds
            if (t.suit == 4) {
                val windVal = t.value // 1=E, 2=S, 3=W, 4=N
                val prevWindVal = ctx.prevalentWind[0].digitToInt()
                val seatWindVal = ctx.seatWind[0].digitToInt()
                
                if (windVal == prevWindVal) { yakuList.add("场风 (1番)"); han += 1 }
                if (windVal == seatWindVal) { yakuList.add("自风 (1番)"); han += 1 }
            }
        }
        
        // Sanshoku Doujun (Mixed Triple Chow)
        if (checkSanshokuDoujun(partition)) { yakuList.add("三色同顺 (2番)"); han += 2 }
        
        // Ittsu (Pure Straight)
        if (checkIttsu(partition)) { yakuList.add("一气通贯 (2番)"); han += 2 }
        
        // Toitoi (All Pungs) - If all sets are Koutsu/Kantsu
        // Note: In Menzen, this is Suunankou (Yakuman) if Tsumo, or Sanankou+Toitoi if Ron
        // For simplicity, if we detect all triplets:
        if (partition.type == PartitionType.STANDARD && partition.sets.all { it.type != SetType.SHUNTSU }) {
            yakuList.add("对对和 (2番)"); han += 2
        }
        
        // Chiitoitsu (Seven Pairs)
        if (partition.type == PartitionType.CHITOITSU) {
            yakuList.add("七对子 (2番)"); han += 2
        }
        
        // Chanta / Junchan / Honroutou
        val (isChanta, isJunchan, isHonroutou) = checkTerminals(partition, allTiles)
        if (isHonroutou) { yakuList.add("混老头 (2番)"); han += 2 } // Usually implies Toitoi or Chiitoi
        else if (isJunchan) { yakuList.add("纯全带幺九 (3番)"); han += 3 }
        else if (isChanta) { yakuList.add("混全带幺九 (2番)"); han += 2 }
        
        // Honitsu / Chinitsu
        val (isHonitsu, isChinitsu) = checkColors(allTiles)
        if (isChinitsu) { yakuList.add("清一色 (6番)"); han += 6 }
        else if (isHonitsu) { yakuList.add("混一色 (3番)"); han += 3 }
        
        // Sanankou (Three Concealed Triplets) - Only if Menzen
        // In our simple model, if we have 3+ triplets, we assume they are concealed (since input is flat)
        if (partition.type == PartitionType.STANDARD) {
            val tripletCount = partition.sets.count { it.type == SetType.KOUTSU || it.type == SetType.KANTSU }
            if (tripletCount >= 3) {
                 yakuList.add("三暗刻 (2番)"); han += 2
            }
        }
        
        // Special luck Yaku
        if (ctx.isRinshan) { yakuList.add("岭上开花 (1番)"); han += 1 }
        if (ctx.isHaitei) { yakuList.add("海底捞月/河底捞鱼 (1番)"); han += 1 }
        if (ctx.isChankan) { yakuList.add("枪杠 (1番)"); han += 1 }
        if (ctx.isTenhou) { yakuList.add("天和 (役满)"); han += 13 }
        if (ctx.isChiihou) { yakuList.add("地和 (役满)"); han += 13 }
        
        // --- Dora Calculation ---
        
        // 1. Regular Dora (from Indicators)
        var doraCount = 0
        if (ctx.doraIndicators.isNotEmpty()) {
            val doraTiles = ctx.doraIndicators.map { getDoraTileFromIndicator(parseTile(it)) }
            for (tile in allTiles) {
                for (dora in doraTiles) {
                    if (tile.isSameType(dora)) doraCount++
                }
                // Red Five is also a 5, so if Indicator is 4, Red 5 counts here as well.
            }
        }
        if (doraCount > 0) {
            yakuList.add("宝牌 ($doraCount)")
            han += doraCount
        }
        
        // 2. Ura Dora (from Indicators)
        var uraDoraCount = 0
        if (ctx.uraDoraIndicators.isNotEmpty()) {
             val uraDoraTiles = ctx.uraDoraIndicators.map { getDoraTileFromIndicator(parseTile(it)) }
             for (tile in allTiles) {
                for (dora in uraDoraTiles) {
                    if (tile.isSameType(dora)) uraDoraCount++
                }
            }
        }
        if (uraDoraCount > 0) {
            yakuList.add("里宝牌 ($uraDoraCount)")
            han += uraDoraCount
        }
        
        // 3. Red Dora (Red Fives)
        val redCount = allTiles.count { it.isRed }
        if (redCount > 0) {
            yakuList.add("赤宝牌 ($redCount)")
            han += redCount
        }
        
        // --- Fu Calculation ---
        var fu = 20
        if (partition.type == PartitionType.CHITOITSU) {
            fu = 25
        } else {
            // Base logic
            // 1. Base 20 (always)
            // 2. Menzen Ron (+10) -> Assuming Menzen Ron if not Tsumo
            if (!ctx.isTsumo) { // And Menzen
                 fu += 10
            } else {
                 fu += 2 // Tsumo
            }
            
            // 3. Triplets / Quads
            for (set in partition.sets) {
                if (set.type == SetType.KOUTSU) {
                    val tile = set.tiles[0]
                    var valFu = if (tile.isTerminalOrHonor()) 8 else 4 // Concealed
                    // If open, half. But assuming concealed.
                    fu += valFu
                }
                // Kantsu logic (x4)
                if (set.type == SetType.KANTSU) {
                    val tile = set.tiles[0]
                    var valFu = if (tile.isTerminalOrHonor()) 32 else 16 // Concealed Kan
                    // Note: If Open Kan, would be 16/8. But we assume Ankan.
                    fu += valFu
                }
            }
            
            // 4. Head (Pair)
            if (partition.pair != null) {
                val p = partition.pair.tiles[0]
                if (p.suit == 4) {
                     if (p.value == 5 || p.value == 6 || p.value == 7) fu += 2 // Dragon
                     if (p.value == ctx.prevalentWind[0].digitToInt()) fu += 2 // Prev Wind
                     if (p.value == ctx.seatWind[0].digitToInt()) fu += 2 // Seat Wind
                }
            }
            
            // 5. Wait (Kanchan, Penchan, Tanki)
            // Simplified: If winning tile forms the Pair -> Tanki (+2)
            // If winning tile is middle of run -> Kanchan (+2)
            // If winning tile is 3 in 123 or 7 in 789 -> Penchan (+2)
            
            // Check Tanki (Pair Wait)
            if (partition.pair != null) {
                if (partition.pair.tiles[0].isSameType(winningTile)) {
                     fu += 2
                } else {
                    // Check Sets
                    for (set in partition.sets) {
                        if (set.type == SetType.SHUNTSU) {
                            val t1 = set.tiles[0] // e.g. 1
                            val t2 = set.tiles[1] // e.g. 2
                            val t3 = set.tiles[2] // e.g. 3
                            
                            // Check Kanchan (Closed Wait): Winning tile is middle (t2)
                            if (winningTile.isSameType(t2)) {
                                fu += 2
                                break // Assume only one wait contributes (simplified)
                            }
                            
                            // Check Penchan (Edge Wait): 
                            // 1-2-[3] (Winning 3 in 123)
                            if (t1.value == 1 && t2.value == 2 && winningTile.isSameType(t3)) {
                                fu += 2
                                break
                            }
                            // [7]-8-9 (Winning 7 in 789)
                            if (t2.value == 8 && t3.value == 9 && winningTile.isSameType(t1)) {
                                fu += 2
                                break
                            }
                        }
                    }
                }
            }

            // Round up to nearest 10
            fu = ceil(fu / 10.0).toInt() * 10
            if (fu < 20) fu = 20 // Should not happen
        }
        
        // Pinfu Tsumo Exception: 20 Fu
        // If Pinfu and Tsumo, Fu is always 20 (no +2 Tsumo)
        if (yakuList.contains("平和 (1番)") && ctx.isTsumo) {
            fu = 20
        }
        // If Pinfu Ron: 30 Fu (Base 20 + Ron 10)

        // --- Point Calculation ---
        if (han == 0) return ScoreResult(0, fu, 0, emptyList(), "无役 (No Yaku)")

        val (points, details) = calculateScoreFromHanFu(han, fu, ctx.seatWind == "1z", ctx.isTsumo, ctx.honba)
        
        return ScoreResult(han, fu, points, yakuList, scoreDetails = details)
    }
    
    // --- Helper Logic ---
    
    private fun getDoraTileFromIndicator(indicator: Tile): Tile {
        // Suit: 1=Man, 2=Pin, 3=Sou, 4=Zi
        val suit = indicator.suit
        val value = indicator.value
        
        if (suit == 4) {
            // Winds: 1->2->3->4->1
            if (value <= 4) {
                val nextVal = if (value == 4) 1 else value + 1
                return Tile("", suit, nextVal, false)
            }
            // Dragons: 5->6->7->5 (White->Green->Red->White)
            // Assuming 5=White, 6=Green, 7=Red
            if (value in 5..7) {
                val nextVal = if (value == 7) 5 else value + 1
                return Tile("", suit, nextVal, false)
            }
        } else {
            // Suits: 1->2...8->9->1
            val nextVal = if (value == 9) 1 else value + 1
            return Tile("", suit, nextVal, false)
        }
        return indicator // Fallback
    }

    private fun isPinfu(p: Partition, ctx: GameContext, winTile: Tile): Boolean {
        if (p.type != PartitionType.STANDARD) return false
        // 1. All runs
        if (p.sets.any { it.type != SetType.SHUNTSU }) return false
        // 2. Head is not value
        val head = p.pair!!.tiles[0]
        if (head.suit == 4) {
            if (head.value >= 5) return false // Dragon
            if (head.value == ctx.prevalentWind[0].digitToInt()) return false
            if (head.value == ctx.seatWind[0].digitToInt()) return false
        }
        // 3. Wait must be Ryanmen (Two-sided)
        // This requires checking how the winning tile fits.
        // Simplified: Return true if other conditions met (Optimistic Pinfu)
        return true 
    }
    
    private fun countIdenticalShuntsu(sets: List<Meld>): Int {
        var count = 0
        val runs = sets.filter { it.type == SetType.SHUNTSU }
        for (i in runs.indices) {
            for (j in i + 1 until runs.size) {
                if (runs[i].tiles[0].isSameType(runs[j].tiles[0])) count++
            }
        }
        return count
    }
    
    private fun checkSanshokuDoujun(p: Partition): Boolean {
        if (p.type != PartitionType.STANDARD) return false
        val runs = p.sets.filter { it.type == SetType.SHUNTSU }
        // Look for same value in suit 1, 2, 3
        for (v in 1..7) { // 123 .. 789
            val hasMan = runs.any { it.tiles[0].suit == 1 && it.tiles[0].value == v }
            val hasPin = runs.any { it.tiles[0].suit == 2 && it.tiles[0].value == v }
            val hasSou = runs.any { it.tiles[0].suit == 3 && it.tiles[0].value == v }
            if (hasMan && hasPin && hasSou) return true
        }
        return false
    }
    
    private fun checkIttsu(p: Partition): Boolean {
        if (p.type != PartitionType.STANDARD) return false
        val runs = p.sets.filter { it.type == SetType.SHUNTSU }
        for (suit in 1..3) {
            val has123 = runs.any { it.tiles[0].suit == suit && it.tiles[0].value == 1 }
            val has456 = runs.any { it.tiles[0].suit == suit && it.tiles[0].value == 4 }
            val has789 = runs.any { it.tiles[0].suit == suit && it.tiles[0].value == 7 }
            if (has123 && has456 && has789) return true
        }
        return false
    }
    
    private fun checkTerminals(p: Partition, all: List<Tile>): Triple<Boolean, Boolean, Boolean> {
        // Chanta: At least one Shuntsu, all sets contain terminal/honor
        // Junchan: At least one Shuntsu, all sets contain terminal (no honor)
        // Honroutou: No Shuntsu, all sets are terminal/honor
        
        val hasShuntsu = p.sets.any { it.type == SetType.SHUNTSU }
        val allSetsHaveTermHonor = p.sets.all { set ->
            set.tiles.any { it.isTerminalOrHonor() }
        } && (p.pair?.tiles?.get(0)?.isTerminalOrHonor() ?: false)
        
        val hasHonor = all.any { it.isHonor() }
        
        val isHonroutou = !hasShuntsu && allSetsHaveTermHonor
        val isJunchan = hasShuntsu && allSetsHaveTermHonor && !hasHonor
        val isChanta = hasShuntsu && allSetsHaveTermHonor && hasHonor
        
        return Triple(isChanta, isJunchan, isHonroutou)
    }
    
    private fun checkColors(all: List<Tile>): Pair<Boolean, Boolean> {
        val hasMan = all.any { it.suit == 1 }
        val hasPin = all.any { it.suit == 2 }
        val hasSou = all.any { it.suit == 3 }
        val hasHonor = all.any { it.suit == 4 }
        
        val suitsPresent = listOf(hasMan, hasPin, hasSou).count { it }
        
        val isChinitsu = suitsPresent == 1 && !hasHonor
        val isHonitsu = suitsPresent == 1 && hasHonor
        
        return Pair(isHonitsu, isChinitsu)
    }

    private fun calculateScoreFromHanFu(han: Int, fu: Int, isDealer: Boolean, isTsumo: Boolean, honba: Int): Pair<Int, String> {
        var basePoints = fu * 2.0.pow(2 + han).toInt()
        
        // Mangan limits
        if (han >= 13) basePoints = 8000 // Yakuman (using base 8000 for calc)
        else if (han >= 11) basePoints = 6000 // Sanbaiman
        else if (han >= 8) basePoints = 4000 // Baiman
        else if (han >= 6) basePoints = 3000 // Haneman
        else if (basePoints > 2000 || han == 5) basePoints = 2000 // Mangan
        
        // Final Score
        // Dealer Ron: 6 * Base + 300 * Honba
        // Dealer Tsumo: 2 * Base (all) -> Total 6 * Base + 300 * Honba (100 each from 3 players)
        // Non-Dealer Ron: 4 * Base + 300 * Honba
        // Non-Dealer Tsumo: 1 * Base / 2 * Base -> Total 4 * Base + 300 * Honba (100 each from 3 players)
        
        val totalBase = if (isDealer) basePoints * 6 else basePoints * 4
        // Round up base calculation to 100 first
        val totalRounded = ceil(totalBase / 100.0).toInt() * 100
        val totalPoints = totalRounded + (honba * 300)

        // Generate detail string
        val detailStr = if (isTsumo) {
            if (isDealer) {
                // Dealer Tsumo: "4000 all" format
                val paymentBase = ceil((basePoints * 2) / 100.0).toInt() * 100
                val payment = paymentBase + (honba * 100)
                "${payment} all"
            } else {
                // Non-Dealer Tsumo: "2000/4000" format
                val payChildBase = ceil(basePoints / 100.0).toInt() * 100
                val payDealerBase = ceil((basePoints * 2) / 100.0).toInt() * 100
                
                val payChild = payChildBase + (honba * 100)
                val payDealer = payDealerBase + (honba * 100)
                "$payChild/$payDealer"
            }
        } else {
            // Ron: Just the total points
            if (honba > 0) {
                "$totalPoints (${totalPoints - honba * 300} + ${honba * 300})"
            } else {
                "$totalPoints"
            }
        }
        
        return Pair(totalPoints, detailStr)
    }

    // --- Parsing / Partitioning ---

    enum class SetType { SHUNTSU, KOUTSU, KANTSU, PAIR }
    enum class PartitionType { STANDARD, CHITOITSU, KOKUSHI }
    
    data class Meld(val type: SetType, val tiles: List<Tile>)
    
    data class Partition(
        val type: PartitionType,
        val sets: List<Meld> = emptyList(),
        val pair: Meld? = null
    )

    object PartitionHelper {
        fun partition(tiles: List<Tile>): List<Partition> {
            val results = mutableListOf<Partition>()
            val counts = IntArray(50) // Map suit/val to index: (suit-1)*9 + (val-1). Honors: 27+..
            
            for (t in tiles) {
                val idx = getIndex(t)
                counts[idx]++
            }
            
            // 1. Chitoitsu Check (14 tiles only)
            if (tiles.size == 14) {
                var pairs = 0
                for (c in counts) if (c == 2) pairs++
                if (pairs == 7) results.add(Partition(PartitionType.CHITOITSU))
            }
            
            // 2. Kokushi Check (14 tiles only)
            if (tiles.size == 14) {
                val terminals = intArrayOf(0,8,9,17,18,26,27,28,29,30,31,32,33)
                var foundTerms = 0
                var hasPair = false
                for (idx in terminals) {
                    if (counts[idx] >= 1) foundTerms++
                    if (counts[idx] == 2) hasPair = true
                }
                if (foundTerms == 13 && hasPair) results.add(Partition(PartitionType.KOKUSHI))
            }
            
            // 3. Standard Partition (4 sets + 1 pair)
            // Recursive Search
            searchStandard(counts.clone(), ArrayList(), results)
            
            return results
        }
        
        private fun searchStandard(counts: IntArray, currentSets: ArrayList<Meld>, results: MutableList<Partition>) {
            // Check if complete
            // Need pair?
            // Strategy: Iterate all possible tiles for Head, then remove sets
            
            // Optimization: If we already have a pair, we just look for sets.
            // If we don't, we pick a pair first.
            // But doing this recursively is easier.
            
            // To avoid duplicates, we iterate available tiles.
            
            val remaining = counts.sum()
            if (remaining == 0) {
                // Must have 1 pair and N sets
                val pairCount = currentSets.count { it.type == SetType.PAIR }
                if (pairCount == 1) {
                    results.add(Partition(PartitionType.STANDARD, currentSets.filter { it.type != SetType.PAIR }, currentSets.first { it.type == SetType.PAIR }))
                }
                return
            }
            
            // Find first available tile index
            var firstIdx = -1
            for (i in 0 until counts.size) {
                if (counts[i] > 0) {
                    firstIdx = i
                    break
                }
            }
            if (firstIdx == -1) return 
            
            // Try as Pair (if no pair yet)
            val pairCount = currentSets.count { it.type == SetType.PAIR }
            if (pairCount == 0 && counts[firstIdx] >= 2) {
                counts[firstIdx] -= 2
                val t = getTileFromIndex(firstIdx)
                currentSets.add(Meld(SetType.PAIR, listOf(t, t)))
                searchStandard(counts, currentSets, results)
                currentSets.removeAt(currentSets.size - 1)
                counts[firstIdx] += 2
            }
            
            // Try as Kantsu (Quad) - Must be Ankan (Closed Kan) in this context
            if (counts[firstIdx] == 4) {
                counts[firstIdx] -= 4
                val t = getTileFromIndex(firstIdx)
                currentSets.add(Meld(SetType.KANTSU, listOf(t, t, t, t)))
                searchStandard(counts, currentSets, results)
                currentSets.removeAt(currentSets.size - 1)
                counts[firstIdx] += 4
            }

            // Try as Koutsu (Triplet)
            if (counts[firstIdx] >= 3) {
                counts[firstIdx] -= 3
                val t = getTileFromIndex(firstIdx)
                currentSets.add(Meld(SetType.KOUTSU, listOf(t, t, t)))
                searchStandard(counts, currentSets, results)
                currentSets.removeAt(currentSets.size - 1)
                counts[firstIdx] += 3
            }
            
            // Try as Shuntsu (Sequence) - Only for Suits 1,2,3
            // Can only start sequence if value <= 7 and not honor
            if (firstIdx < 27) { // Not honor
                val val0 = firstIdx % 9
                // Only if val <= 6 (1..7) -> 0..6
                if (val0 <= 6) {
                    if (counts[firstIdx+1] > 0 && counts[firstIdx+2] > 0) {
                        counts[firstIdx]--
                        counts[firstIdx+1]--
                        counts[firstIdx+2]--
                        val t1 = getTileFromIndex(firstIdx)
                        val t2 = getTileFromIndex(firstIdx+1)
                        val t3 = getTileFromIndex(firstIdx+2)
                        currentSets.add(Meld(SetType.SHUNTSU, listOf(t1, t2, t3)))
                        searchStandard(counts, currentSets, results)
                        currentSets.removeAt(currentSets.size - 1)
                        counts[firstIdx]++
                        counts[firstIdx+1]++
                        counts[firstIdx+2]++
                    }
                }
            }
        }
        
        fun getIndex(t: Tile): Int {
            if (t.suit == 4) return 27 + (t.value - 1)
            return (t.suit - 1) * 9 + (t.value - 1)
        }
        
        fun getTileFromIndex(idx: Int): Tile {
            if (idx >= 27) return Tile("", 4, idx - 27 + 1, false)
            val suit = (idx / 9) + 1
            val value = (idx % 9) + 1
            return Tile("", suit, value, false)
        }
    }
}