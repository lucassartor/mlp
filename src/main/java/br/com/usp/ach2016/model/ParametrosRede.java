package br.com.usp.ach2016.model;

public record ParametrosRede(
        int tamanhoEntrada,
        int tamanhoCamadaEscondida,
        int tamanhoSaida,
        long sementeAleatoria,
        String nomeExecucao
) {}
