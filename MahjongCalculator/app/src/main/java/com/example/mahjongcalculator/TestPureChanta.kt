package com.example.mahjongcalculator

fun main() {
    val calculator = MahjongCalculator()
    
    // Case: 1s1s1s 2s2s2s 3s3s3s 7s8s9s 9s(Winning)
    // This should be interpreted as three 123s sequences (Pure Straight / Pinfu)
    // rather than three triplets (Sanankou).
    
    val tiles = listOf(
        "1s", "1s", "1s",
        "2s", "2s", "2s",
        "3s", "3s", "3s",
        "7s", "8s", "9s",
        "9s", "9s" 
    )
    val winningTile = "9s"
    
    val context = MahjongCalculator.GameContext(
        prevalentWind = "1z", // East
        seatWind = "1z",      // East
        doraIndicators = listOf("4m"), // Irrelevant
        isTsumo = true,
        isRiichi = false,
        honba = 0
    )
    
    val result = calculator.calculate(tiles, winningTile, context)
    
    println("--- Calculation Result ---")
    println("Points: ${result.points}")
    println("Han: ${result.han}")
    println("Fu: ${result.fu}")
    println("Yaku: ${result.yakuList}")
    println("Error: ${result.error}")
}
