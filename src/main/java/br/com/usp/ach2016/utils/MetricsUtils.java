package br.com.usp.ach2016.utils;

import org.ejml.simple.SimpleMatrix;

import java.util.concurrent.TimeUnit;

public class MetricsUtils {

    public static double calcularAcuracia(SimpleMatrix yVerdadeiroOneHot, SimpleMatrix yPrevistoProbs) {
        if (yVerdadeiroOneHot.getNumRows() != yPrevistoProbs.getNumRows() || yVerdadeiroOneHot.getNumCols() != yPrevistoProbs.getNumCols()) {
            throw new IllegalArgumentException("Dimensões das matrizes de rótulos e previsões devem ser iguais para calcular acurácia.");
        }
        int numAmostras = yVerdadeiroOneHot.getNumRows();
        if (numAmostras == 0) {
            return 0.0; // Evita divisão por zero
        }

        int predicoesCorretas = 0;
        for (int i = 0; i < numAmostras; i++) {
            int indicePrevisto = encontrarIndiceMaximo(yPrevistoProbs.getRow(i));
            int indiceReal = encontrarIndiceMaximo(yVerdadeiroOneHot.getRow(i));

            if (indicePrevisto == indiceReal && indiceReal != -1) {
                predicoesCorretas++;
            }
        }

        return (double) predicoesCorretas / numAmostras * 100.0;
    }

    // --- Método Auxiliar para Encontrar Índice do Máximo em um Vetor Linha (ou índice do 1.0) ---
    private static int encontrarIndiceMaximo(SimpleMatrix vetorLinha) {
        int indiceMax = -1;
        double valorMax = -Double.MAX_VALUE;
        for (int j = 0; j < vetorLinha.getNumCols(); j++) {
            if (vetorLinha.get(0, j) > valorMax) {
                valorMax = vetorLinha.get(0, j);
                indiceMax = j;
            }
        }
        return indiceMax;
    }

    /**
     * Formata a duração em milissegundos para uma string legível (HH:mm:ss.SSS).
     */
    public static String formatarDuracao(long millis) {
        long horas = TimeUnit.MILLISECONDS.toHours(millis);
        millis -= TimeUnit.HOURS.toMillis(horas);
        long minutos = TimeUnit.MILLISECONDS.toMinutes(millis);
        millis -= TimeUnit.MINUTES.toMillis(minutos);
        long segundos = TimeUnit.MILLISECONDS.toSeconds(millis);
        millis -= TimeUnit.SECONDS.toMillis(segundos);
        return String.format("%02d:%02d:%02d.%03d", horas, minutos, segundos, millis);
    }
}
