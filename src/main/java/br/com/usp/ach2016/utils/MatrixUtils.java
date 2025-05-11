package br.com.usp.ach2016.utils;

import br.com.usp.ach2016.model.Dataset;
import br.com.usp.ach2016.model.ResultadoAnaliseConfusao;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

import static br.com.usp.ach2016.utils.MetricsUtils.calcularAcuracia;

public class MatrixUtils {

    // todo: talvez tirar daqui
    public static ResultadoAnaliseConfusao gerarResultadosClassificacao(int[][] matrizConfusao,
                                                                        Dataset dataset,
                                                                        SimpleMatrix previsoesTeste) {
        int numClasses = previsoesTeste.getNumCols();
        int[] tps = new int[numClasses];
        int[] fps = new int[numClasses];
        int[] fns = new int[numClasses];

        int numTotalAmostras = previsoesTeste.getNumRows();
        double acuraciaTeste = calcularAcuracia(dataset.yTeste(), previsoesTeste);
        int numPredicoesCorretas = (int) Math.round((acuraciaTeste / 100.0) * numTotalAmostras);

        for (int i = 0; i < numClasses; i++) {
            tps[i] = matrizConfusao[i][i];
            int somaLinhaI = 0;
            int somaColunaI = 0;
            for (int j = 0; j < numClasses; j++) {
                somaLinhaI += matrizConfusao[i][j]; // Soma da linha i (Real = i)
                somaColunaI += matrizConfusao[j][i]; // Soma da coluna i (Previsto = i)
            }
            fns[i] = somaLinhaI - tps[i]; // Tudo na linha i que não é TP é FN para a classe i
            fps[i] = somaColunaI - tps[i]; // Tudo na coluna i que não é TP é FP para a classe i
        }

        return new ResultadoAnaliseConfusao(matrizConfusao, tps, fps, fns, numTotalAmostras, numPredicoesCorretas, acuraciaTeste);
    }

    /**
     * Calcula a matriz de confusão para avaliar o desempenho da classificação.
     *
     * @param yVerdadeiroOneHot Matriz dos rótulos verdadeiros no formato one-hot.
     * @param yPrevistoProbs Matriz das saídas da rede (probabilidades ou ativações).
     * @return Uma matriz 2D de inteiros representando a matriz de confusão (linhas=Real, colunas=Previsto).
     */
    public static int[][] calcularMatrizConfusao(SimpleMatrix yVerdadeiroOneHot, SimpleMatrix yPrevistoProbs) {
        int numClasses = yVerdadeiroOneHot.getNumCols();
        int numAmostras = yVerdadeiroOneHot.getNumRows();
        int[][] matrizConfusao = new int[numClasses][numClasses];

        // Itera sobre cada amostra
        for (int i = 0; i < numAmostras; i++) {
            // Encontra o índice da classe verdadeira (onde está o 1.0 no vetor one-hot)
            int indiceReal = -1;
            for (int j = 0; j < numClasses; j++) {
                if (yVerdadeiroOneHot.get(i, j) == 1.0) {
                    indiceReal = j;
                    break;
                }
            }

            // Encontra o índice da classe prevista (índice do maior valor na saída da rede)
            int indicePrevisto = -1;
            double valorMaximo = -Double.MAX_VALUE; // Começa com o menor valor possível
            for (int j = 0; j < numClasses; j++) {
                if (yPrevistoProbs.get(i, j) > valorMaximo) {
                    valorMaximo = yPrevistoProbs.get(i, j);
                    indicePrevisto = j;
                }
            }

            // Incrementa a célula correspondente na matriz de confusão se ambos os índices foram encontrados
            if (indiceReal != -1 && indicePrevisto != -1) {
                matrizConfusao[indiceReal][indicePrevisto]++;
            } else {
                System.err.println("Aviso: Não foi possível determinar índice real ou previsto para a amostra " + i + " na matriz de confusão.");
            }
        }
        return matrizConfusao;
    }

    // --- Método Auxiliar para Combinar Matrizes Verticalmente (para k-fold) ---
    public static SimpleMatrix combinarMatrizesVerticalmente(SimpleMatrix mat1, SimpleMatrix mat2) {
        if (mat1 == null && mat2 == null) return null;
        if (mat1 == null) return mat2.copy();
        if (mat2 == null) return mat1.copy();

        if (mat1.getNumCols() != mat2.getNumCols()) {
            throw new IllegalArgumentException("Matrizes devem ter o mesmo número de colunas para combinar verticalmente.");
        }

        SimpleMatrix combinada = new SimpleMatrix(mat1.getNumRows() + mat2.getNumRows(), mat1.getNumCols());
        CommonOps_DDRM.extract(mat1.getDDRM(), 0, mat1.getNumRows(), 0, mat1.getNumCols(), combinada.getDDRM(), 0, 0);
        CommonOps_DDRM.extract(mat2.getDDRM(), 0, mat2.getNumRows(), 0, mat2.getNumCols(), combinada.getDDRM(), mat1.getNumRows(), 0);
        return combinada;
    }
}
