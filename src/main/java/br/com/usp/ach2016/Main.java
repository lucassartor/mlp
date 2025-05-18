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

import java.util.Random;

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
        //System.out.println("\n\n===== EXECUCAO 1: TREINO SIMPLES (SEM VALIDACAO/PARADA ANTECIPADA) =====");
        execucaoSimples.executar();

        //batchExecutar(dataset);

        // --- Execução 2: Treino COM Parada Antecipada ---
        //Execucao execucaoComParadaAntecipada = getExecucaoComParadaAntecipada(dataset);
        //System.out.println("\n\n===== EXECUCAO 2: TREINO COM PARADA ANTECIPADA =====");
        //execucaoComParadaAntecipada.executar();

        // --- Execução 3: Demonstração da Validação Cruzada (k-Fold) ---
        // TODO: Arrumar
        //Execucao execucaoComValidacaoCruzada = getExecucaoComValidacaoCruzada(dataset);
        //System.out.println("\n\n===== EXECUCAO 3: TREINO COM VALIDACAO CRUZADA =====");
        //execucaoComValidacaoCruzada.executar();

        System.out.println("\n\nTodas as execucoes foram concluidas.");
    }

    public static void executarBatchExploracaoAleatoria(Dataset dataset) {
        System.out.println("\n\n===== INICIANDO BATCH DE EXPLORAÇÃO ALEATÓRIA DE HIPERPARÂMETROS =====");
        System.out.println("Pressione Ctrl+C para interromper o batch a qualquer momento.");

        Random seletorHiperparametros = new Random(); // Semente não fixa para variar a cada execução do Main

        // Defina os intervalos e opções para os hiperparâmetros
        int[] opcoesHiddenSize = {25, 50, 75, 100, 125, 150, 200};
        double[] opcoesTaxaAprendizado = {0.75, 1.0, 1.25, 1.50, 1.75};
        int minEpocas = 100;
        int maxEpocas = 70000; // Limite superior para épocas

        while (true) {
            // Selecionar hiperparâmetros aleatoriamente
            int hiddenSizeAtual = opcoesHiddenSize[seletorHiperparametros.nextInt(opcoesHiddenSize.length)];
            double taxaAprendizadoAtual = opcoesTaxaAprendizado[seletorHiperparametros.nextInt(opcoesTaxaAprendizado.length)];
            // Gera épocas aleatórias entre minEpocas e maxEpocas, em múltiplos de (ex) 1000
            int epocasAtuais = seletorHiperparametros.nextInt((maxEpocas - minEpocas) / 1000 + 1) * 1000;


            ParametrosRede configRede = new ParametrosRede(
                    dataset.xTreino().getNumCols(), hiddenSizeAtual, dataset.yTreino().getNumCols(),
                    SEMENTE_ALEATORIA, // Semente fixa para a MLP, para reprodutibilidade DA MLP
                    NOME_PROBLEMA + "_Simples"
            );
            ParametrosTreinamento configTreino = new ParametrosTreinamento(
                    taxaAprendizadoAtual, epocasAtuais, 0
            );

            Execucao execucaoSimples = new ExecucaoSimples(dataset, configRede, configTreino);
            execucaoSimples.executar();
        }
    }

        private static void batchExecutar(Dataset dataset) {

            // Hiperparâmetros fixos para este batch (exceto LR e épocas para certos testes)
            final int hiddenSizeFixo = 50;

            // Cenários de Taxa de Aprendizado
            int[] hiddenSizes = {5, 25, 50, 75, 100};
            int[] epocasParaTesteLR = {1000, 5000, 10000, 30000}; // Testar cada LR com poucas e muitas épocas

            for (int hiddenSize : hiddenSizes) {
                for (int epocasAtuais : epocasParaTesteLR) {

                ParametrosRede parametrosRedeSimples = new ParametrosRede(
                        dataset.xTreino().getNumCols(),
                        hiddenSize, // hiddenSize
                        dataset.yTreino().getNumCols(),
                        SEMENTE_ALEATORIA,
                        NOME_PROBLEMA + "_Simples"
                );

                ParametrosTreinamento parametrosTreinamentoSimples = new ParametrosTreinamento(
                        5.00,
                        epocasAtuais,
                        0
                );

                Execucao execucaoSimples = new ExecucaoSimples(dataset, parametrosRedeSimples, parametrosTreinamentoSimples);
                execucaoSimples.executar();
            }
        }
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
                1.25,    // taxaAprendizado
                5000,  // epocas
                0
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
                0.5,
                500,
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
                0.5,
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