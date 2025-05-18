package br.com.usp.ach2016.utils;

import br.com.usp.ach2016.model.Dataset;
import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Classe utilitária para carregamento e pré-processamento de conjuntos de dados.
 * Focada no carregamento do conjunto de dados de caracteres
 */
public class DataUtils {

    /**
     * Carrega o conjunto de dados de caracteres a partir dos arquivos de entrada (X) e rótulos (Y).
     * Os arquivos são lidos como recursos do classpath.
     * Os pixels em X são convertidos de -1/1 para 0.0/1.0.
     * Os rótulos de caracteres em Y são convertidos para o formato one-hot encoding.
     * Os dados são divididos em conjuntos de treinamento, validação e teste.
     *
     * @param caminhoX Caminho para o arquivo X.txt dentro da pasta resources (ex: "datasets/caracteres/X.txt").
     * @param caminhoY Caminho para o arquivo Y_letra.txt dentro da pasta resources (ex: "datasets/caracteres/Y_letra.txt").
     * @param numAmostrasTeste Número de amostras a serem separadas para o conjunto de teste (as últimas N amostras do arquivo).
     * @param numAmostrasValidacao Número de amostras (retiradas do final do conjunto de treino original) para o conjunto de validação
     * @return Um objeto `Dataset` contendo as matrizes de treino/teste e informações de classe, ou `null` se ocorrer um erro.
     */
    public static Dataset carregarDadosDataset(String caminhoX,
                                               String caminhoY,
                                               int numAmostrasTeste,
                                               int numAmostrasValidacao) {
        List<String> linhasYBrutas = new ArrayList<>();
        List<double[]> todasLinhasX = new ArrayList<>();
        List<Character> rotulosClasses = new ArrayList<>();
        Map<Character, Integer> caractereParaIndice = new HashMap<>();

        InputStream fluxoEntradaY = DataUtils.class.getClassLoader().getResourceAsStream(caminhoY);
        if (fluxoEntradaY == null) {
            System.err.println("Erro Crítico: Arquivo de rótulos Y não encontrado no classpath: " + caminhoY);
            return null;
        }
        try (BufferedReader leitorY = new BufferedReader(new InputStreamReader(fluxoEntradaY))) {
            String linha;
            Set<Character> caracteresUnicos = new HashSet<>(); // Usado para encontrar todas as classes únicas
            boolean primeiraLinha = true;
            while ((linha = leitorY.readLine()) != null) {
                // Trata o BOM (Byte Order Mark) que pode estar presente na primeira linha
                if (primeiraLinha && !linha.isEmpty() && linha.charAt(0) == '\uFEFF') {
                    linha = linha.substring(1); // Remove o BOM
                }
                primeiraLinha = false;

                String rotuloStr = linha.trim();
                if (!rotuloStr.isEmpty()) {
                    linhasYBrutas.add(rotuloStr);
                    char rotuloChar = rotuloStr.charAt(0);
                    caracteresUnicos.add(rotuloChar);
                }
            }
            rotulosClasses.addAll(caracteresUnicos);
            Collections.sort(rotulosClasses);
            for (int i = 0; i < rotulosClasses.size(); i++) {
                caractereParaIndice.put(rotulosClasses.get(i), i);
            }
            System.out.println("Classes identificadas (" + rotulosClasses.size() + "): " + rotulosClasses);

        } catch (IOException e) {
            System.err.println("Erro de IO ao ler o arquivo de rótulos Y do classpath: " + caminhoY);
            e.printStackTrace();
            return null;
        }

        int numTotalAmostras = linhasYBrutas.size();
        int numClasses = rotulosClasses.size();
        int numAtributosEntrada = 120; // Definido pela especificação (10x12 pixels)

        if (numTotalAmostras == 0 || numClasses == 0) {
            System.err.println("Erro: Nenhum rótulo ou classe válida encontrada no arquivo Y.");
            return null;
        }

        InputStream fluxoEntradaX = DataUtils.class.getClassLoader().getResourceAsStream(caminhoX);
        if (fluxoEntradaX == null) {
            System.err.println("Erro: Arquivo de entradas X não encontrado no classpath: " + caminhoX);
            return null;
        }
        try (BufferedReader leitorX = new BufferedReader(new InputStreamReader(fluxoEntradaX))) {
            String linha;
            boolean primeiraLinhaX = true;
            int contadorLinha = 0;
            while ((linha = leitorX.readLine()) != null) {
                contadorLinha++;
                // Trata BOM
                if (primeiraLinhaX && !linha.isEmpty() && linha.charAt(0) == '\uFEFF') {
                    linha = linha.substring(1);
                }
                primeiraLinhaX = false;
                if (linha.trim().isEmpty()) continue;

                // Divide a linha por vírgula, permitindo espaços ao redor da vírgula
                String[] valoresPixelStr = linha.trim().split("\\s*,\\s*");

                // Remove o último elemento se estiver vazio (caso haja uma vírgula no final da linha)
                if (valoresPixelStr.length > 0 && valoresPixelStr[valoresPixelStr.length - 1].isEmpty()) {
                    valoresPixelStr = Arrays.copyOf(valoresPixelStr, valoresPixelStr.length - 1);
                }

                if (valoresPixelStr.length != numAtributosEntrada) {
                    System.err.println("Conteúdo da linha problemática: [" + linha.trim() + "]");
                    throw new IOException("Inconsistência no número de colunas na linha " + contadorLinha + " do arquivo X.");
                }

                // Converte a string de pixels para double[] (0.0 ou 1.0)
                double[] linhaPixel = new double[numAtributosEntrada];
                for (int i = 0; i < numAtributosEntrada; i++) {
                    // Converte o valor lido ("-1" ou "1") para double e depois mapeia para 0.0 ou 1.0
                    linhaPixel[i] = (Double.parseDouble(valoresPixelStr[i].trim()) + 1.0) / 2.0;
                }
                todasLinhasX.add(linhaPixel);
            }
        } catch (IOException e) {
            System.err.println("Erro de IO ao ler o arquivo de entradas X do classpath: " + caminhoX);
            e.printStackTrace();
            return null;
        }

        if (todasLinhasX.size() != numTotalAmostras) {
            System.err.println("Erro: Número final de amostras em X (" + todasLinhasX.size() +
                    ") não coincide com Y (" + numTotalAmostras + "). Verifique os arquivos de dados.");
            return null;
        }

        List<double[]> todasLinhasYOneHot = new ArrayList<>();
        for (String rotuloStr : linhasYBrutas) {
            char rotuloChar = rotuloStr.charAt(0);
            Integer indiceClasse = caractereParaIndice.get(rotuloChar);
            double[] vetorOneHot = new double[numClasses]; // Cria vetor de zeros
            vetorOneHot[indiceClasse] = 1.0; // Define 1.0 na posição correta
            todasLinhasYOneHot.add(vetorOneHot);
        }

        SimpleMatrix X_completo = new SimpleMatrix(numTotalAmostras, numAtributosEntrada);
        SimpleMatrix Y_completo_one_hot = new SimpleMatrix(numTotalAmostras, numClasses);

        for (int i = 0; i < numTotalAmostras; i++) {
            X_completo.setRow(i, 0, todasLinhasX.get(i));
            Y_completo_one_hot.setRow(i, 0, todasLinhasYOneHot.get(i));
        }

        if (numAmostrasTeste + numAmostrasValidacao >= numTotalAmostras || numAmostrasTeste < 0 || numAmostrasValidacao < 0) {
            System.err.println("Erro: Número inválido de amostras de teste (" + numAmostrasTeste +
                    ") ou validacao (" + numAmostrasValidacao + ") para o total de amostras (" + numTotalAmostras + ").");
            return null;
        }

        int indiceFimTreino = numTotalAmostras - numAmostrasTeste - numAmostrasValidacao;
        int indiceFimValidacao = numTotalAmostras - numAmostrasTeste;

        SimpleMatrix xTreino = X_completo.extractMatrix(0, indiceFimTreino, 0, X_completo.getNumCols());
        SimpleMatrix yTreino = Y_completo_one_hot.extractMatrix(0, indiceFimTreino, 0, Y_completo_one_hot.getNumCols());

        SimpleMatrix xValidacao = X_completo.extractMatrix(indiceFimTreino, indiceFimValidacao, 0, X_completo.getNumCols());
        SimpleMatrix yValidacao = Y_completo_one_hot.extractMatrix(indiceFimTreino, indiceFimValidacao, 0, Y_completo_one_hot.getNumCols());

        SimpleMatrix xTeste = X_completo.extractMatrix(indiceFimValidacao, numTotalAmostras, 0, X_completo.getNumCols());
        SimpleMatrix yTeste = Y_completo_one_hot.extractMatrix(indiceFimValidacao, numTotalAmostras, 0, Y_completo_one_hot.getNumCols());

        System.out.println("Dados carregados e divididos:");
        System.out.println("xTreino:    " + xTreino.getNumRows() + "x" + xTreino.getNumCols());
        System.out.println("yTreino:    " + yTreino.getNumRows() + "x" + yTreino.getNumCols());
        System.out.println("xValidacao: " + xValidacao.getNumRows() + "x" + xValidacao.getNumCols());
        System.out.println("yValidacao: " + yValidacao.getNumRows() + "x" + yValidacao.getNumCols());
        System.out.println("xTeste:     " + xTeste.getNumRows() + "x" + xTeste.getNumCols());
        System.out.println("yTeste:     " + yTeste.getNumRows() + "x" + yTeste.getNumCols());

        return new Dataset(xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste, rotulosClasses, caractereParaIndice);
    }
}