package br.com.usp.ach2016;

import static br.com.usp.ach2016.Constants.*;
import static br.com.usp.ach2016.utils.DataUtils.*;
import static br.com.usp.ach2016.utils.IOUtils.*;

import br.com.usp.ach2016.executions.Execucao;
import br.com.usp.ach2016.executions.ExecucaoComParadaAntecipada;
import br.com.usp.ach2016.executions.ExecucaoSimples;
import br.com.usp.ach2016.executions.ExecucaoValidacaoCruzada;
import br.com.usp.ach2016.model.Dataset;
import br.com.usp.ach2016.model.ParametrosRede;
import br.com.usp.ach2016.model.ParametrosTreinamento;

/**
 * Classe que orquestra a execução do EP1 - MLP.
 */
public class Main {

    public static void main(String[] args) {
        criarPastaResultados();

        // --- Carregamento dos Dados ---
        System.out.println("\n--- Carregando Dados para " + NOME_PROBLEMA + " ---");
        Dataset dataset = carregarDadosDataset(CAMINHO_X, CAMINHO_Y, NUM_AMOSTRAS_TESTE, NUM_AMOSTRAS_VALIDACAO);
        if (dataset == null) {
            System.err.println("Falha ao carregar dados. Encerrando.");
            return;
        }

        // --- Execução 1: Treino Simples (Sem Validação Cruzada, Sem Parada Antecipada) ---
        Execucao execucaoSimples = getExecucaoSimples(dataset);
        System.out.println("\n\n===== EXECUCAO 1: TREINO SIMPLES (SEM VALIDACAO/PARADA ANTECIPADA) =====");
        execucaoSimples.executar();

        // --- Execução 2: Treino COM Parada Antecipada ---
        Execucao execucaoComParadaAntecipada = getExecucaoComParadaAntecipada(dataset);
        System.out.println("\n\n===== EXECUCAO 2: TREINO COM PARADA ANTECIPADA =====");
        execucaoComParadaAntecipada.executar();

        // --- Execução 3: Demonstração da Validação Cruzada (k-Fold) ---
        // TODO: Arrumar
        Execucao execucaoComValidacaoCruzada = getExecucaoComValidacaoCruzada(dataset);
        System.out.println("\n\n===== EXECUCAO 3: TREINO COM VALIDACAO CRUZADA =====");
        execucaoComValidacaoCruzada.executar();

        System.out.println("\n\nTodas as execucoes foram concluidas.");
    }

    private static ExecucaoSimples getExecucaoSimples(Dataset dataset) {
        ParametrosRede parametrosRedeSimples = new ParametrosRede(
                dataset.xTreino().getNumCols(),
                50, // hiddenSize
                dataset.yTreino().getNumCols(),
                SEMENTE_ALEATORIA,
                NOME_PROBLEMA + "_Simples"
        );
        ParametrosTreinamento parametrosTreinamentoSimples = new ParametrosTreinamento(
                0.05,    // taxaAprendizado
                1000,  // epocas
                0      // paciencia (0 desativa parada antecipada)
        );
        return new ExecucaoSimples(dataset, parametrosRedeSimples, parametrosTreinamentoSimples);
    }


    private static ExecucaoComParadaAntecipada getExecucaoComParadaAntecipada(Dataset dataset) {
        ParametrosRede parametrosRedeParadaAntecipada = new ParametrosRede(
                dataset.xTreino().getNumCols(),
                50,
                dataset.yTreino().getNumCols(),
                SEMENTE_ALEATORIA,
                NOME_PROBLEMA + "_ParadaAntecipada"
        );
        ParametrosTreinamento parametrosTreinamentoParadaAntecipada = new ParametrosTreinamento(
                0.05,
                20000,
                1000
        );
        return new ExecucaoComParadaAntecipada(dataset, parametrosRedeParadaAntecipada, parametrosTreinamentoParadaAntecipada);
    }

    private static ExecucaoValidacaoCruzada getExecucaoComValidacaoCruzada(Dataset dataset) {
        ParametrosRede parametrosRedeValidacaoCruzada = new ParametrosRede(
                dataset.xTreino().getNumCols(),
                50,
                dataset.yTreino().getNumCols(),
                SEMENTE_ALEATORIA,
                NOME_PROBLEMA + "_ValidacaoCruzada"
        );
        ParametrosTreinamento parametrosTreinamentoValidacaoCruzada = new ParametrosTreinamento(
                0.05,
                30000, // epocas por k fold
                0
        );
        int kFolds = 5;
        return new ExecucaoValidacaoCruzada(
                dataset,
                parametrosRedeValidacaoCruzada,
                parametrosTreinamentoValidacaoCruzada,
                kFolds
        );
    }
}