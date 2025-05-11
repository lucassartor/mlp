package br.com.usp.ach2016.executions;

import br.com.usp.ach2016.MLP;
import br.com.usp.ach2016.model.Dataset;
import br.com.usp.ach2016.model.ParametrosRede;
import br.com.usp.ach2016.model.ParametrosTreinamento;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

import static br.com.usp.ach2016.utils.MatrixUtils.combinarMatrizesVerticalmente;
import static br.com.usp.ach2016.utils.MetricsUtils.calcularAcuracia;

public class ExecucaoValidacaoCruzada extends Execucao {

    public final int kFolds;

    public ExecucaoValidacaoCruzada(Dataset dataset,
                                    ParametrosRede parametrosRede,
                                    ParametrosTreinamento parametrosTreinamento,
                                    int kFolds) {
        super(dataset, parametrosRede, parametrosTreinamento);
        this.kFolds = kFolds;
    }

    @Override
    public void executar() {
        System.out.println("\n\n===== INICIANDO: " + parametrosRede.nomeExecucao().toUpperCase() + " =====");

        // Combina treino e validação originais para formar o pool de dados do k-fold
        SimpleMatrix xTreinoVal = combinarMatrizesVerticalmente(dataset.xTreino(), dataset.xValidacao());
        SimpleMatrix yTreinoVal = combinarMatrizesVerticalmente(dataset.yTreino(), dataset.yValidacao());

        int numTotalAmostrasTreinoVal = xTreinoVal.getNumRows();
        if (numTotalAmostrasTreinoVal < kFolds) {
            System.err.println("Erro: Número de folds (k=" + kFolds + ") é maior que o número de amostras (" + numTotalAmostrasTreinoVal + ")");
            return;
        }
        int tamanhoFold = numTotalAmostrasTreinoVal / kFolds;
        int resto = numTotalAmostrasTreinoVal % kFolds;

        System.out.println("Total de amostras para k-fold: " + numTotalAmostrasTreinoVal);
        System.out.println("Tamanho aproximado de cada fold: " + tamanhoFold + " (com ajuste para resto)");

        List<Double> acuraciasValidacaoFolds = new ArrayList<>();
        List<Double> errosFinaisTreinoFolds = new ArrayList<>();

        int indiceInicioFold = 0;
        for (int foldAtual = 0; foldAtual < kFolds; foldAtual++) {
            System.out.println("\n--- Fold " + (foldAtual + 1) + "/" + kFolds + " ---");

            // Define o tamanho e os índices deste fold de validação específico
            int tamanhoFoldAtual = tamanhoFold + (foldAtual < resto ? 1 : 0); // Distribui o resto
            int indiceFimFold = indiceInicioFold + tamanhoFoldAtual;

            // Separa dados de validação e treino para este fold
            SimpleMatrix xValidacaoFold = xTreinoVal.extractMatrix(indiceInicioFold, indiceFimFold, 0, xTreinoVal.getNumCols());
            SimpleMatrix yValidacaoFold = yTreinoVal.extractMatrix(indiceInicioFold, indiceFimFold, 0, yTreinoVal.getNumCols());

            SimpleMatrix xTreinoFoldParte1 = indiceInicioFold == 0 ? null : xTreinoVal.extractMatrix(0, indiceInicioFold, 0, xTreinoVal.getNumCols());
            SimpleMatrix yTreinoFoldParte1 = indiceInicioFold == 0 ? null : yTreinoVal.extractMatrix(0, indiceInicioFold, 0, yTreinoVal.getNumCols());

            SimpleMatrix xTreinoFoldParte2 = indiceFimFold == numTotalAmostrasTreinoVal ? null : xTreinoVal.extractMatrix(indiceFimFold, numTotalAmostrasTreinoVal, 0, xTreinoVal.getNumCols());
            SimpleMatrix yTreinoFoldParte2 = indiceFimFold == numTotalAmostrasTreinoVal ? null : yTreinoVal.extractMatrix(indiceFimFold, numTotalAmostrasTreinoVal, 0, yTreinoVal.getNumCols());

            // Combina as partes de treino (se houver duas partes)
            SimpleMatrix xTreinoFold = combinarMatrizesVerticalmente(xTreinoFoldParte1, xTreinoFoldParte2);
            SimpleMatrix yTreinoFold = combinarMatrizesVerticalmente(yTreinoFoldParte1, yTreinoFoldParte2);

            System.out.println("Tamanho Treino Fold " + (foldAtual+1) + ": " + xTreinoFold.getNumRows() + "x" + xTreinoFold.getNumCols());
            System.out.println("Tamanho Validacao Fold " + (foldAtual+1) + ": " + xValidacaoFold.getNumRows() + "x" + xValidacaoFold.getNumCols());

            // Cria e treina uma NOVA rede para este fold
            String pastKFold = caminhoPastaResultadosExecucao.substring("resultados/".length()) + "/Fold" + (foldAtual + 1);
            ParametrosRede parametrosRedeFold = new ParametrosRede(
                    parametrosRede.tamanhoEntrada(),
                    parametrosRede.tamanhoCamadaEscondida(),
                    parametrosRede.tamanhoSaida(),
                    parametrosRede.sementeAleatoria() + foldAtual, // Muda a semente por fold
                    pastKFold);
            MLP redeFold = new MLP(parametrosRedeFold);

            // Treina por um número FIXO de épocas (sem parada antecipada aqui para simplificar a demo)
            redeFold.treinar(xTreinoFold, yTreinoFold, null, null, parametrosTreinamento);
            errosFinaisTreinoFolds.add(redeFold.historicoErro.get(redeFold.historicoErro.size()-1));

            // Avalia no conjunto de validação deste fold
            SimpleMatrix previsoesValidacaoFold = redeFold.prever(xValidacaoFold);
            double acuraciaFold = calcularAcuracia(yValidacaoFold, previsoesValidacaoFold);
            acuraciasValidacaoFolds.add(acuraciaFold);
            System.out.printf("Fold %d - Acuracia Validacao: %.2f%%\n", foldAtual + 1, acuraciaFold);

            indiceInicioFold = indiceFimFold; // Atualiza o índice para o próximo fold
        }

        // Calcula e exibe a média das acurácias de validação
        double somaAcuracias = 0;
        for(double acc : acuraciasValidacaoFolds) somaAcuracias += acc;
        double mediaAcuraciaValidacao = somaAcuracias / kFolds;
        System.out.printf("\nResultado Validacao Cruzada (k=%d): Acurácia Media = %.2f%%\n", kFolds, mediaAcuraciaValidacao);

        System.out.println("===== FINALIZADA: " + parametrosRede.nomeExecucao().toUpperCase() + " =====");
    }
}
