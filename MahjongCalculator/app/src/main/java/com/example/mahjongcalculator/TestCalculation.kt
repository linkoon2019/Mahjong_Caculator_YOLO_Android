package com.example.mahjongcalculator

fun main() {
    val calculator = MahjongCalculator()
    
    // 1p x3, 2p x3, 3p x3, 7s, 8s, 9s, 9s(winning)
    // Note: The input list to calculate includes ALL tiles (14 total)
    // "牌型是1p1p1p2p2p2p3p3p3p7s8s9s9s" (13 tiles) + "自摸9s"
    val tiles = listOf(
        "1p", "1p", "1p",
        "2p", "2p", "2p",
        "3p", "3p", "3p",
        "7s", "8s", "9s",
        "9s", "9s" // 9s pair + winning 9s
    )
    val winningTile = "9s"
    
    val context = MahjongCalculator.GameContext(
        prevalentWind = "1z", // East
        seatWind = "1z",      // East
        doraIndicators = listOf("4m"),
        isTsumo = true,
        isRiichi = false,
        honba = 0
    )
    
    val result = calculator.calculate(tiles, winningTile, context)
    
    println("Points: ${result.points}")
    println("Han: ${result.han}")
    println("Fu: ${result.fu}")
    println("Yaku: ${result.yakuList}")
}
