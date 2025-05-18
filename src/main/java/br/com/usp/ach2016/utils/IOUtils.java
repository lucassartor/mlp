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

    private static String criarDiretorioSeNaoExistir(String caminhoDiretorio, boolean flag) {
        File dir = new File(caminhoDiretorio);
        if (!dir.exists()) {
            boolean success = dir.mkdirs();
            if (success) {
                return caminhoDiretorio;  // Retorna o diretório criado
            } else {
                return getString(caminhoDiretorio);
            }
        }
        return getString(caminhoDiretorio);
    }

    private static String getString(String caminhoDiretorio) {
        if (caminhoDiretorio.contains("execucao")) {
            // Se já existe, tenta incrementar o sufixo numérico
            String base = caminhoDiretorio;
            int numero = 2;

            if (caminhoDiretorio.matches(".*\\d+$")) {
                int i = caminhoDiretorio.length() - 1;
                while (i >= 0 && Character.isDigit(caminhoDiretorio.charAt(i))) {
                    i--;
                }
                base = caminhoDiretorio.substring(0, i + 1);
                numero = Integer.parseInt(caminhoDiretorio.substring(i + 1)) + 1;
            }

            while (true) {
                String novoCaminho = base + numero;
                File novoDir = new File(novoCaminho);
                if (!novoDir.exists()) {
                    return criarDiretorioSeNaoExistir(novoCaminho, true);  // recursivo e retorna o nome criado
                }
                numero++;
            }
        }

        return caminhoDiretorio;
    }

    public static void criarPastaResultados() {
        criarDiretorioSeNaoExistir(DIRETORIO_SAIDA);
    }

    public static String criarPastaResultadosExecucao(String nomeExecucao) {
        return criarDiretorioSeNaoExistir(DIRETORIO_SAIDA + "/" + nomeExecucao, true);
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
                escritor.printf(Locale.US, "%.8f\n", erro);
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

            escritor.println("========================================================");
            escritor.println("        HIPERPARÂMETROS E CONFIGURAÇÃO DA EXECUÇÃO");
            escritor.println("========================================================");
            escritor.println();

            escritor.println("--- Configuração do Dataset ---");
            escritor.println("Dataset Utilizado: CARACTERES");
            escritor.println();

            escritor.println("--- Arquitetura da Rede Neural (MLP) ---");
            escritor.println("Tamanho da Camada de Entrada: " + parametrosRede.tamanhoEntrada());
            escritor.println("Tamanho da Camada Escondida: " + parametrosRede.tamanhoCamadaEscondida());
            escritor.println("Tamanho da Camada de Saída: " + parametrosRede.tamanhoSaida());
            escritor.println("Função de Ativação (Escondida e Saída): Sigmoid");
            escritor.println();

            escritor.println("--- Parâmetros de Inicialização e Treinamento ---");
            escritor.println("Semente Aleatória para Pesos (Random Seed): " + parametrosRede.sementeAleatoria());
            escritor.println("Taxa de Aprendizado (Learning Rate): " + String.format(Locale.US, "%.4f", parametrosTreinamento.taxaAprendizado()));
            escritor.println("Número de Épocas Alvo/Máximo: " + parametrosTreinamento.epocas());
            escritor.println("Otimizador: Gradiente Descendente em Lote (Batch Gradient Descent)");

            if (parametrosTreinamento.pacienciaParadaAntecipada() > 0) {
                escritor.println("Parada Antecipada Ativada: Sim");
                escritor.println("Paciência para Parada Antecipada: " + parametrosTreinamento.pacienciaParadaAntecipada() + " épocas");
            } else {
                escritor.println("Parada Antecipada Ativada: Não");
            }
            escritor.println();

            System.out.println("Hiperparâmetros salvos em " + caminhoArquivo);

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

            // --- IMPRIMIR A MATRIZ DE CONFUSÃO FORMATADA ---
            escritor.println("--- Matriz de Confusão (Conjunto de Treino) ---");
            escritor.println("(Linhas = Real, Colunas = Previsto)");

            // Cabeçalho da Matriz (Classes Previstas)
            escritor.print("Real\\Prev"); // Canto superior esquerdo
            for (Character rotulo : rotulosClasses) {
                escritor.printf("%5c", rotulo); // Alinha cada rótulo de classe em 5 espaços
            }
            escritor.println();

            // Linhas da Matriz
            int[][] matrizConfusao = analiseConfusao.matrizConfusao();
            for (int i = 0; i < matrizConfusao.length; i++) {
                escritor.printf("%-9c", rotulosClasses.get(i)); // Rótulo da classe real (alinhado à esquerda em 5 espaços)
                for (int j = 0; j < matrizConfusao[i].length; j++) {
                    escritor.printf("%5d", matrizConfusao[i][j]); // Alinha cada contagem em 5 espaços
                }
                escritor.println();
            }
            escritor.println("-----------------------------------------------------------------------"); // Linha separadora

        } catch (IOException e) {
            System.err.println("Erro ao salvar o relatório detalhado: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void salvarHistoricoDeltaPesos(String caminhoPastaDaExecucao, List<Double> historicoDelta, String nomeCamada) {
        String caminhoArquivo = caminhoPastaDaExecucao + "/historico_delta_" + nomeCamada + ".csv";
        System.out.println("Salvando histórico de delta de pesos para " + nomeCamada + " em: " + caminhoArquivo);
        try (PrintWriter escritor = new PrintWriter(new FileWriter(caminhoArquivo))) {
            escritor.println("Epoca,Media_Magnitude_Delta_" + nomeCamada);
            for (int i = 0; i < historicoDelta.size(); i++) {
                escritor.printf(Locale.US, "%d,%.8e\n", i + 1, historicoDelta.get(i)); // %.8e para notação científica se os deltas ficarem mto pequenos
            }
            System.out.println("Histórico de delta de pesos salvo.");
        } catch (IOException e) {
            System.err.println("Erro ao salvar histórico de delta de pesos para " + nomeCamada + ": " + e.getMessage());
        }
    }
}
