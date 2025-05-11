package br.com.usp.ach2016.model;

public record ParametrosTreinamento(
        double taxaAprendizado,
        int epocas,
        int pacienciaParadaAntecipada
) {}