package br.com.usp.ach2016.model;

public record ResultadoAnaliseConfusao(
        int[][] matrizConfusao,
        int[] verdadeirosPositivos, // TP por classe
        int[] falsosPositivos,    // FP por classe
        int[] falsosNegativos,     // FN por classe
        int numTotalAmostras,
        int numPredicoesCorretas,
        double acuracia
) {}