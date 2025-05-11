package br.com.usp.ach2016.utils;

import br.com.usp.ach2016.MLP;
import br.com.usp.ach2016.model.Dataset;
import br.com.usp.ach2016.model.ParametrosRede;
import br.com.usp.ach2016.model.ParametrosTreinamento;
import br.com.usp.ach2016.model.ResultadoAnaliseConfusao;
import org.ejml.simple.SimpleMatrix;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class IOUtils {

    static final String DIRETORIO_SAIDA = "resultados";

    private static void criarDiretorioSeNaoExistir(String caminhoDiretorio) {
        File dir = new File(caminhoDiretorio);
        if (!dir.exists()) {
            boolean success = dir.mkdirs();
            if (success) {
                System.out.println("Diretório criado: " + caminhoDiretorio);
            } else {
                System.err.println("Falha ao criar diretório: " + caminhoDiretorio);
            }
        }
    }

    public static void criarPastaResultados() {
        criarDiretorioSeNaoExistir(DIRETORIO_SAIDA);
    }

    public static void criarPastaResultadosExecucao(String nomeExecucao) {
        criarDiretorioSeNaoExistir(DIRETORIO_SAIDA + "/" + nomeExecucao);
    }

    public static String criarPastaResultadosNExecucao(String nomeExecucao) {
        String caminhoBase = DIRETORIO_SAIDA + "/" + nomeExecucao;

        int contadorExecucao = 1;
        Path caminhoExecucao;
        while (true) {
            caminhoExecucao = Paths.get(caminhoBase, "execucao" + contadorExecucao);
            if (!Files.exists(caminhoExecucao)) {
                break;
            }
            contadorExecucao++;
        }
        criarDiretorioSeNaoExistir(caminhoExecucao.toString());

        return caminhoExecucao.toString();
    }

    /**
     * Salva os pesos e biases iniciais em arquivos CSV no diretório de saída.
     */
    public static void salvarPesosIniciais(final String caminhoPastaDaExecucao,
                                           final SimpleMatrix pesosIniciaisW1,
                                           final SimpleMatrix biasInicialB1,
                                           final SimpleMatrix pesosIniciaisW2,
                                           final SimpleMatrix biasInicialB2) {
        try {
            // Salva cada matriz/vetor em um arquivo CSV separado.
            pesosIniciaisW1.saveToFileCSV(caminhoPastaDaExecucao + "/pesos_iniciais_W1.csv");
            biasInicialB1.saveToFileCSV(caminhoPastaDaExecucao + "/bias_inicial_b1.csv");
            pesosIniciaisW2.saveToFileCSV(caminhoPastaDaExecucao + "/pesos_iniciais_W2.csv");
            biasInicialB2.saveToFileCSV(caminhoPastaDaExecucao+ "/bias_inicial_b2.csv");
        } catch (IOException e) {
            System.err.println("Erro ao salvar pesos iniciais: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Salva os pesos e biases FINAIS (após o treinamento) em arquivos CSV.
     */
    public static void salvarPesosFinais(final String caminhoPastaDaExecucao,
                                         final SimpleMatrix W1,
                                         final SimpleMatrix b1,
                                         final SimpleMatrix W2,
                                         final SimpleMatrix b2) {
        try {
            W1.saveToFileCSV(caminhoPastaDaExecucao + "/pesos_finais_W1.csv");
            b1.saveToFileCSV(caminhoPastaDaExecucao + "/bias_final_b1.csv");
            W2.saveToFileCSV(caminhoPastaDaExecucao + "/pesos_finais_W2.csv");
            b2.saveToFileCSV(caminhoPastaDaExecucao + "/bias_final_b2.csv");
        } catch (IOException e) {
            System.err.println("Erro ao salvar pesos finais: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void salvarResultadosTreino(final String caminhoPastaDaExecucao,
                                              final ParametrosTreinamento parametrosTreinamento,
                                              final MLP mlp) {
        ParametrosRede parametrosRede = mlp.parametrosRede;
        System.out.println("--- Salvando Resultados do Treino (" + parametrosRede.nomeExecucao() + ") ---");
        salvarHiperparametros(caminhoPastaDaExecucao, parametrosTreinamento, parametrosRede);
        salvarPesosFinais(caminhoPastaDaExecucao, mlp.W1, mlp.b1, mlp.W2, mlp.b2);
        salvarHistoricoErro(caminhoPastaDaExecucao, mlp.historicoErro);
    }

    public static void salvarResultadosTeste(final String caminhoPastaDaExecucao,
                                             final Dataset dataset,
                                             final SimpleMatrix saidasYPrevistas,
                                             final ParametrosRede parametrosRede,
                                             final ParametrosTreinamento parametrosTreinamento,
                                             Map<String, Object> paramsAdicionaisTreino,
                                             final ResultadoAnaliseConfusao resultadoAnaliseConfusao) {
        System.out.println("--- Salvando Resultados do Teste (" + parametrosRede.nomeExecucao() + ") ---");

        salvarSaidasTeste(caminhoPastaDaExecucao, dataset.xTeste(), saidasYPrevistas);

        salvarRelatorioTesteDetalhado(
                caminhoPastaDaExecucao,
                parametrosRede,
                parametrosTreinamento,
                paramsAdicionaisTreino,
                resultadoAnaliseConfusao,
                dataset.rotulosClasses()
        );
    }

    /**
     * Salva o histórico de erros (MSE por época) em um arquivo CSV.
     *
     */
    public static void salvarHistoricoErro(final String caminhoPastaDaExecucao,
                                           final List<Double> historicoErro) {
        String caminhoArquivo = caminhoPastaDaExecucao + "/historico_erro.csv";
        try (PrintWriter escritor = new PrintWriter(new FileWriter(caminhoArquivo))) {
            escritor.println("Erro_Epoca");
            for (Double erro : historicoErro) {
                escritor.printf("%.8f\n", erro);
            }
        } catch (IOException e) {
            System.err.println("Erro ao salvar histórico de erros: " + e.getMessage());
            e.printStackTrace();
        }
    }


    public static void salvarHiperparametros(final String caminhoPastaDaExecucao,
                                             final ParametrosTreinamento parametrosTreinamento,
                                             final ParametrosRede parametrosRede) {
        String caminhoArquivo = caminhoPastaDaExecucao + "/hiperparametros.txt";
        try (PrintWriter escritor = new PrintWriter(new FileWriter(caminhoArquivo))) {
            escritor.println("--- Hiperparâmetros e Arquitetura da Rede ---");
            escritor.println("Tamanho da Entrada (Input Size): " + parametrosRede.tamanhoEntrada());
            escritor.println("Tamanho da Camada Escondida (Hidden Size): " + parametrosRede.tamanhoCamadaEscondida());
            escritor.println("Tamanho da Saída (Output Size): " + parametrosRede.tamanhoSaida());
            escritor.println("Semente Aleatória (Random Seed): " + parametrosRede.sementeAleatoria());
            escritor.println("Função de Ativação (Camada Escondida e Saída): Sigmoid");

            escritor.println("\n--- Hiperparâmetros de Treinamento ---");
            escritor.println("Épocas: " + parametrosTreinamento.epocas());
            escritor.println("Taxa de Aprendizado (Learning Rate): " + parametrosTreinamento.taxaAprendizado());
            if (parametrosTreinamento.pacienciaParadaAntecipada() > 0) {
                escritor.println("Paciencia da Parada Antecipada: " + parametrosTreinamento.pacienciaParadaAntecipada());
            }
        } catch (IOException e) {
            System.err.println("Erro ao salvar hiperparâmetros: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Realiza previsões no conjunto de teste e salva as entradas e as previsões correspondentes
     * em um arquivo CSV.
     *
     */
    public static void salvarSaidasTeste(final String caminhoPastaDaExecucao,
                                         final SimpleMatrix dadosXTeste,
                                         final SimpleMatrix saidasYPrevistas) {
        String caminhoArquivo = caminhoPastaDaExecucao + "/saidas_teste.csv";
        if (dadosXTeste.getNumRows() != saidasYPrevistas.getNumRows()) {
            System.err.println("Erro: Número de linhas das entradas de teste e saídas previstas não coincidem ao salvar.");
            return;
        }

        try (PrintWriter escritor = new PrintWriter(new FileWriter(caminhoArquivo))) {
            // Cria o cabeçalho do CSV
            StringBuilder cabecalho = new StringBuilder();
            for (int i = 0; i < dadosXTeste.getNumCols(); i++) {
                cabecalho.append("Entrada_").append(i + 1).append(",");
            }
            for (int i = 0; i < saidasYPrevistas.getNumCols(); i++) {
                cabecalho.append("Previsao_").append(i + 1).append(i == saidasYPrevistas.getNumCols() - 1 ? "" : ",");
            }
            escritor.println(cabecalho.toString());

            for (int i = 0; i < dadosXTeste.getNumRows(); i++) { // Itera sobre as amostras
                StringBuilder linha = new StringBuilder();
                for (int j = 0; j < dadosXTeste.getNumCols(); j++) {
                    linha.append(String.format(Locale.US, "%.8f", dadosXTeste.get(i, j))).append(","); // Usar Locale.US para ponto decimal
                }
                for (int j = 0; j < saidasYPrevistas.getNumCols(); j++) {
                    linha.append(String.format(Locale.US, "%.8f", saidasYPrevistas.get(i, j))); // Usar Locale.US para ponto decimal
                    if (j < saidasYPrevistas.getNumCols() - 1) {
                        linha.append(",");
                    }
                }
                escritor.println(linha.toString());
            }
        } catch (IOException e) {
            System.err.println("Erro ao salvar saídas de teste: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Salva a matriz de confusão em um arquivo CSV formatado.
     *
     */
    public static void salvarMatrizConfusao(final String caminhoPastaDaExecucao,
                                            final int[][] matrizConfusao,
                                            final List<Character> rotulosClasses) {
        String caminhoArquivo = caminhoPastaDaExecucao + "/matriz_confusao.csv";
        try (PrintWriter escritor = new PrintWriter(new FileWriter(caminhoArquivo))) {
            escritor.print("Real\\Previsto");
            for (Character rotulo : rotulosClasses) {
                escritor.print("," + rotulo);
            }
            escritor.println();

            // Escreve as linhas da matriz
            for (int i = 0; i < matrizConfusao.length; i++) {
                escritor.print(rotulosClasses.get(i));
                for (int j = 0; j < matrizConfusao[i].length; j++) {
                    escritor.print("," + matrizConfusao[i][j]);
                }
                escritor.println();
            }
        } catch (IOException e) {
            System.err.println("Erro ao salvar a matriz de confusão: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Salva um relatório detalhado dos resultados do teste em um arquivo de texto.
     *
     * @param caminhoPastaDaExecucao caminhoPastaDaExecucao
     * @param parametrosRede Configurações da arquitetura da rede.
     * @param parametrosTreinamento Configurações do treinamento.
     * @param paramsAdicionaisTreino Parâmetros adicionais como épocas efetivas, melhor erro de validação.
     * @param analiseConfusao todo
     * @param rotulosClasses Lista dos rótulos das classes.
     */
    public static void salvarRelatorioTesteDetalhado(String caminhoPastaDaExecucao,
                                                     ParametrosRede parametrosRede,
                                                     ParametrosTreinamento parametrosTreinamento,
                                                     Map<String, Object> paramsAdicionaisTreino,
                                                     ResultadoAnaliseConfusao analiseConfusao,
                                                     List<Character> rotulosClasses) {
        String caminhoArquivo = caminhoPastaDaExecucao + "/relatorio_detalhado.txt";

        try (PrintWriter escritor = new PrintWriter(new FileWriter(caminhoArquivo))) {
            SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
            escritor.println("===========================================================");
            escritor.println("        RELATÓRIO DE TESTE - MLP - " + parametrosRede.nomeExecucao());
            escritor.println("===========================================================");
            escritor.println("Data da Geração do Relatório: " + sdf.format(new Date()));
            escritor.println();

            escritor.println("--- Configuração da Rede e Treinamento ---");
            escritor.println("Dataset: CARACTERES COMPLETO");
            escritor.println("Nome da Execução/Treino: " + parametrosRede.nomeExecucao());
            escritor.println("Tamanho da Entrada: " + parametrosRede.tamanhoEntrada());
            escritor.println("Tamanho da Camada Escondida: " + parametrosRede.tamanhoCamadaEscondida());
            escritor.println("Tamanho da Saída: " + parametrosRede.tamanhoSaida());
            escritor.println("Semente Aleatória: " + parametrosRede.sementeAleatoria());
            escritor.println("Taxa de Aprendizado: " + String.format(Locale.US, "%.4f", parametrosTreinamento.taxaAprendizado()));
            escritor.println("Número de Épocas: " + parametrosTreinamento.epocas());
            if (parametrosTreinamento.pacienciaParadaAntecipada() > 0) {
                escritor.println("Paciência Parada Antecipada: " + parametrosTreinamento.pacienciaParadaAntecipada());
            }
            if (paramsAdicionaisTreino != null) {
                paramsAdicionaisTreino.forEach((key, value) -> escritor.println(key + ": " + value));
            }
            escritor.println();

            escritor.println("--- Desempenho de Previsão no Conjunto de Teste ---");
            escritor.println("Total de Amostras: " + analiseConfusao.numTotalAmostras());
            escritor.println("Previsões Corretas: " + analiseConfusao.numPredicoesCorretas());
            escritor.println("Previsões Incorretas: " + (analiseConfusao.numTotalAmostras() - analiseConfusao.numPredicoesCorretas()));
            escritor.println("Acurácia Geral: " + String.format(Locale.US, "%.2f%%", analiseConfusao.acuracia()));
            escritor.println();

            escritor.println("--- Métricas por Classe no Conjunto de Teste ---");
            escritor.println(String.format("%-7s | %4s | %4s | %4s | %10s | %10s | %10s",
                    "Classe", "TP", "FP", "FN", "Precisão", "Revocação", "Medida-F1"));
            escritor.println("-----------------------------------------------------------------------");

            int[] tps = analiseConfusao.verdadeirosPositivos();
            int[] fps = analiseConfusao.falsosPositivos();
            int[] fns = analiseConfusao.falsosNegativos();
            double somaPrecisao = 0, somaRevocacao = 0, somaF1 = 0;
            int classesComAmostras = 0; // Para calcular macro-average corretamente

            for (int i = 0; i < rotulosClasses.size(); i++) {
                char classe = rotulosClasses.get(i);
                double precisao = (tps[i] + fps[i] == 0) ? 0 : (double) tps[i] / (tps[i] + fps[i]);
                double revocacao = (tps[i] + fns[i] == 0) ? 0 : (double) tps[i] / (tps[i] + fns[i]);
                double f1 = (precisao + revocacao == 0) ? 0 : 2 * (precisao * revocacao) / (precisao + revocacao);

                escritor.println(String.format(Locale.US, "%-7c | %4d | %4d | %4d | %9.2f%% | %9.2f%% | %9.2f%%",
                        classe, tps[i], fps[i], fns[i], precisao * 100, revocacao * 100, f1 * 100));

                if (tps[i] + fns[i] > 0) { // Só considera classes presentes no conjunto para a média
                    somaPrecisao += precisao;
                    somaRevocacao += revocacao;
                    somaF1 += f1; // F1 pode ser NaN se precisão e revocação forem 0, tratar se necessário
                    classesComAmostras++;
                }
            }
            escritor.println("-----------------------------------------------------------------------");
            if (classesComAmostras > 0) {
                escritor.println(String.format(Locale.US, "Médias (Macro): Precisão: %.2f%%, Revocação: %.2f%%, Medida-F1: %.2f%%",
                        (somaPrecisao / classesComAmostras) * 100,
                        (somaRevocacao / classesComAmostras) * 100,
                        (somaF1 / classesComAmostras) * 100));
            } else {
                escritor.println("Médias (Macro): N/A (nenhuma classe com amostras encontradas)");
            }
            escritor.println("-----------------------------------------------------------------------");
            escritor.println();

        } catch (IOException e) {
            System.err.println("Erro ao salvar o relatório detalhado: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
