package br.com.usp.ach2016;

import static br.com.usp.ach2016.utils.IOUtils.*;

import br.com.usp.ach2016.model.ParametrosRede;
import br.com.usp.ach2016.model.ParametrosTreinamento;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.*;

/**
 * Implementa uma Rede Neural Artificial Multilayer Perceptron (MLP)
 * com uma camada escondida.
 * Treinada com o algoritmo Backpropagation usando Gradiente Descendente em Lote (Batch Gradient Descent).
 * Utiliza a biblioteca EJML para operações de matrizes.
 */
public class MLP {

    public final ParametrosRede parametrosRede;

    // --- Atributos de Inicialização e Treinamento ---
    private final Random geradorAleatorio; // Gerador de números aleatórios
    public List<Double> historicoErro; // Armazena o erro médio quadrático a cada época de treinamento
    List<Double> historicoErroValidacao;

    // --- Pesos e Biases da Rede ---
    // Notação: W1: Pesos Entrada -> Escondida; b1: Bias Escondida; W2: Pesos Escondida -> Saída; b2: Bias Saída
    public SimpleMatrix W1, b1, W2, b2;

    // Melhores pesos encontrados durante o treino com validação
    private SimpleMatrix melhorW1, melhorB1, melhorW2, melhorB2;
    public double melhorErroValidacao = Double.MAX_VALUE;
    public int epocaMelhorErro = 0;

    // --- Cópias dos Pesos Iniciais (para relatório/análise) ---
    public SimpleMatrix pesosIniciaisW1, biasInicialB1, pesosIniciaisW2, biasInicialB2;

    // --- Gradientes (calculados durante o backpropagation) ---
    private SimpleMatrix gradW1, gradB1, gradW2, gradB2; // Derivadas parciais do Erro em relação aos pesos/biases

    // --- Atributos para Armazenar Valores Intermediários (passo forward/backward) ---
    // Guardam os resultados da última execução do passo forward para serem usados no backward
    private SimpleMatrix ultimaEntradaX; // Entrada X do último passo forward
    private SimpleMatrix z1, a1;         // z1: Soma ponderada + bias da camada escondida; a1: Ativação da camada escondida (pós-sigmoid)
    private SimpleMatrix z2, saidaRede;  // z2: Soma ponderada + bias da camada de saída; saidaRede: Ativação final (pós-sigmoid)

    /**
     * Construtor da Rede MLP.
     * Inicializa a arquitetura, os pesos, biases e prepara para o treinamento.
     *
     * @param parametrosRede Objeto contendo os parâmetros da arquitetura da rede.
     */
    public MLP(ParametrosRede parametrosRede) {
        // 1. Define os tamanhos das camadas
        this.parametrosRede = parametrosRede;
        criarPastaResultadosExecucao(parametrosRede.nomeExecucao());

        // 2. Configura o gerador de números aleatórios
        this.geradorAleatorio = new Random(parametrosRede.sementeAleatoria());

        // 3. Inicializa a lista para guardar o histórico de erros
        this.historicoErro = new ArrayList<>();
        this.historicoErroValidacao = new ArrayList<>();

        // 4. Inicializa os pesos e biases
        inicializarPesosEBias();
    }

    /**
     * Inicializa as matrizes de pesos (W1, W2) e vetores de bias (b1, b2).
     * Pesos são inicializados com valores aleatórios de uma distribuição normal gaussiana (média 0, desvio 1).
     * Biases são inicializados com zeros.
     * Também guarda cópias dos pesos/biases iniciais.
     */
    private void inicializarPesosEBias() {
        // --- Inicialização de W1 (Pesos Entrada -> Escondida) ---
        // Shape: [tamanhoEntrada x tamanhoCamadaEscondida]
        this.W1 = new SimpleMatrix(this.parametrosRede.tamanhoEntrada(), this.parametrosRede.tamanhoCamadaEscondida());
        for (int i = 0; i < this.W1.getNumRows(); i++) {
            for (int j = 0; j < this.W1.getNumCols(); j++) {
                // Preenche com valores da distribuição Gaussiana
                this.W1.set(i, j, this.geradorAleatorio.nextGaussian());
            }
        }

        // --- Inicialização de b1 (Bias Camada Escondida) ---
        // Shape: [1 x tamanhoCamadaEscondida] (Vetor Linha)
        // SimpleMatrix inicializa com zeros por padrão.
        this.b1 = new SimpleMatrix(1, this.parametrosRede.tamanhoCamadaEscondida());

        // --- Inicialização de W2 (Pesos Escondida -> Saída) ---
        // Shape: [tamanhoCamadaEscondida x tamanhoSaida]
        this.W2 = new SimpleMatrix(this.parametrosRede.tamanhoCamadaEscondida(), this.parametrosRede.tamanhoSaida());
        for (int i = 0; i < this.W2.getNumRows(); i++) {
            for (int j = 0; j < this.W2.getNumCols(); j++) {
                this.W2.set(i, j, this.geradorAleatorio.nextGaussian());
            }
        }

        // --- Inicialização de b2 (Bias Camada Saída) ---
        // Shape: [1 x tamanhoSaida] (Vetor Linha)
        this.b2 = new SimpleMatrix(1, this.parametrosRede.tamanhoSaida());

        // --- Guardar Cópias Iniciais ---
        this.pesosIniciaisW1 = this.W1.copy();
        this.biasInicialB1 = this.b1.copy();
        this.pesosIniciaisW2 = this.W2.copy();
        this.biasInicialB2 = this.b2.copy();
    }

    /**
     * Calcula a função de ativação Sigmoid para cada elemento da matriz de entrada.
     * sigmoid(x) = 1 / (1 + exp(-x))
     * Inclui clipping para evitar problemas numéricos com exp().
     *
     * @param x Matriz de entrada (geralmente a soma ponderada + bias, 'z').
     * @return Uma nova matriz com a função Sigmoid aplicada a cada elemento.
     */
    private SimpleMatrix sigmoid(SimpleMatrix x) {
        SimpleMatrix resultado = new SimpleMatrix(x.getNumRows(), x.getNumCols());
        for (int r = 0; r < x.getNumRows(); r++) {
            for (int c = 0; c < x.getNumCols(); c++) {
                double valor = x.get(r, c);
                // Clipping: limita o valor de entrada para exp() para evitar Infinity ou 0.0
                valor = Math.max(-500.0, Math.min(500.0, valor));
                resultado.set(r, c, 1.0 / (1.0 + Math.exp(-valor)));
            }
        }
        return resultado;
    }

    /**
     * Calcula a derivada da função Sigmoid.
     * A derivada de sigmoid(z) é sigmoid(z) * (1 - sigmoid(z)).
     * Este método recebe a *saída* da sigmoid (a = sigmoid(z)) e calcula a * (1 - a).
     *
     * @param ativacao A matriz de ativação (o resultado da aplicação da sigmoid).
     * @return Uma nova matriz contendo a derivada da sigmoid calculada para cada elemento.
     */
    private SimpleMatrix derivadaSigmoid(SimpleMatrix ativacao) {
        // Cria uma matriz de 'uns' com as mesmas dimensões da ativação
        SimpleMatrix uns = new SimpleMatrix(ativacao.getNumRows(), ativacao.getNumCols());
        uns.fill(1.0);
        // Calcula 'ativacao * (1 - ativacao)' elemento a elemento
        // uns.minus(ativacao) -> (1 - ativacao)
        // ativacao.elementMult(...) -> multiplica elemento a elemento
        return ativacao.elementMult(uns.minus(ativacao));
    }

    /**
     * Executa o passo de propagação direta (forward pass) da rede neural.
     * Calcula as ativações das camadas escondida e de saída para um lote de entradas.
     * Armazena os resultados intermediários (z1, a1, z2, saidaRede) para uso no backpropagation.
     *
     * @param loteEntradaX Matriz contendo as amostras de entrada (uma amostra por linha). Shape: [numAmostras, tamanhoEntrada].
     * @return A matriz de saída da rede (ativação da última camada). Shape: [numAmostras, tamanhoSaida].
     */
    public SimpleMatrix forward(SimpleMatrix loteEntradaX) {
        // Armazena a entrada atual para ser usada no cálculo dos gradientes no backpropagation
        this.ultimaEntradaX = loteEntradaX;

        // --- Camada Escondida ---
        // 1. Calcular a soma ponderada + bias: z1 = (X @ W1) + b1
        //    X @ W1 -> multiplicação de matrizes
        this.z1 = this.ultimaEntradaX.mult(this.W1); // z1 tem shape [numAmostras, tamanhoCamadaEscondida]

        // Adicionar o bias b1 (vetor linha) a cada linha de z1.
        for (int i = 0; i < this.z1.getNumRows(); i++) {
            DMatrixRMaj linha_i = CommonOps_DDRM.extractRow(this.z1.getDDRM(), i, null); // Extrai linha i
            CommonOps_DDRM.add(linha_i, this.b1.getDDRM(), linha_i); // Soma a linha com b1 (elemento a elemento)
            CommonOps_DDRM.insert(linha_i, this.z1.getDDRM(), i, 0); // Insere a linha modificada de volta em z1
        }

        // 2. Aplicar a função de ativação: a1 = sigmoid(z1)
        this.a1 = sigmoid(this.z1); // a1 tem a mesma shape de z1

        // --- Camada de Saída ---
        // 3. Calcular a soma ponderada + bias: z2 = (a1 @ W2) + b2
        this.z2 = this.a1.mult(this.W2); // z2 tem shape [numAmostras, tamanhoSaida]

        // Adicionar o bias b2
        for (int i = 0; i < this.z2.getNumRows(); i++) {
            DMatrixRMaj linha_i = CommonOps_DDRM.extractRow(this.z2.getDDRM(), i, null);
            CommonOps_DDRM.add(linha_i, this.b2.getDDRM(), linha_i);
            CommonOps_DDRM.insert(linha_i, this.z2.getDDRM(), i, 0);
        }

        // 4. Aplicar a função de ativação: saidaRede = sigmoid(z2)
        this.saidaRede = sigmoid(this.z2); // saidaRede tem a mesma shape de z2

        return this.saidaRede;
    }

    /**
     * Calcula o Erro Quadrático Médio (Mean Squared Error - MSE) entre os rótulos verdadeiros e as previsões da rede.
     * MSE = (1 / numAmostras) * 0.5 * sum( (yVerdadeiro_ij - yPrevisto_ij)^2 )
     *
     * @param yVerdadeiro Matriz com os rótulos verdadeiros (geralmente one-hot).
     * @param yPrevisto   Matriz com as previsões (saídas) da rede.
     * @return O valor do erro quadrático médio para o lote.
     * @throws IllegalArgumentException Se as dimensões de yVerdadeiro e yPrevisto não forem iguais.
     */
    private double erroQuadraticoMedio(SimpleMatrix yVerdadeiro, SimpleMatrix yPrevisto) {
        if (yVerdadeiro.getNumRows() != yPrevisto.getNumRows() || yVerdadeiro.getNumCols() != yPrevisto.getNumCols()) {
            throw new IllegalArgumentException("Matrizes yVerdadeiro e yPrevisto devem ter as mesmas dimensões para calcular o MSE.");
        }

        // Calcula a diferença (erro) elemento a elemento: erro = yVerdadeiro - yPrevisto
        SimpleMatrix matrizErro = yVerdadeiro.minus(yPrevisto);

        // Eleva cada elemento do erro ao quadrado
        SimpleMatrix matrizErroQuadrado = matrizErro.elementPower(2.0);

        // Soma todos os elementos da matriz de erros quadrados
        double somaErrosQuadrados = matrizErroQuadrado.elementSum();

        // Calcula a média: (0.5 * soma_total) / numero_de_amostras
        // A divisão pelo número de amostras faz a média do erro por amostra no lote.
        return (0.5 * somaErrosQuadrados) / yVerdadeiro.getNumRows();
    }

    /**
     * Executa o passo de backpropagation para calcular os gradientes
     * do erro em relação a todos os pesos e biases da rede.
     * Utiliza os valores intermediários armazenados pelo último passo `forward`.
     * Os gradientes calculados são armazenados nos atributos `gradW1`, `gradB1`, `gradW2`, `gradB2`.
     *
     * @param yVerdadeiro Matriz com os rótulos verdadeiros correspondentes à última entrada processada por `forward`.
     * @throws IllegalArgumentException Se as dimensões de yVerdadeiro não baterem com a saída da rede.
     */
    public void backward(SimpleMatrix yVerdadeiro) {
        // Validação das dimensões
        if (yVerdadeiro.getNumRows() != this.saidaRede.getNumRows() || yVerdadeiro.getNumCols() != this.saidaRede.getNumCols()) {
            throw new IllegalArgumentException("Dimensões de yVerdadeiro incompatíveis com a saída da rede no passo backward.");
        }

        int numAmostras = this.ultimaEntradaX.getNumRows(); // Número de amostras no lote processado

        // --- Etapa 1: Calcular Gradientes para a Camada de Saída (W2, b2) ---

        // Erro na saída (diferença entre previsão e valor real)
        // Para a derivada da função de custo MSE (0.5 * (y_true - y_pred)^2), a derivada em relação a y_pred é -(y_true - y_pred) = y_pred - y_true
        // dE/da2 (derivada parcial do Erro Total E em relação à ativação da saída a2)
        SimpleMatrix erroSaida = this.saidaRede.minus(yVerdadeiro);

        // Derivada da função de ativação Sigmoid na saída (calculada em z2, mas usando a2 = saidaRede)
        // da2/dz2 = a2 * (1 - a2)
        SimpleMatrix derivadaSigmoidSaida = derivadaSigmoid(this.saidaRede);

        // Delta da camada de saída (delta_k ou delta_output)
        // delta_output = (erro na saída) * (derivada da ativação na saída) --- multiplicação elemento a elemento
        // delta_output = dE/da2 * da2/dz2 = dE/dz2
        SimpleMatrix deltaSaida = erroSaida.elementMult(derivadaSigmoidSaida); // Shape: [numAmostras, tamanhoSaida]

        // Gradiente para os pesos W2 (dE/dW2)
        // dE/dW2 = (a1^T @ delta_output) / numAmostras (média sobre o lote)
        // Onde a1 é a ativação da camada escondida (armazenada em this.a1)
        // a1^T tem shape [tamanhoCamadaEscondida, numAmostras]
        // delta_output tem shape [numAmostras, tamanhoSaida]
        // O resultado dW2 tem shape [tamanhoCamadaEscondida, tamanhoSaida], igual a W2.
        this.gradW2 = this.a1.transpose().mult(deltaSaida).divide(numAmostras);

        // Gradiente para o bias b2 (dE/db2)
        // dE/db2 = sum(delta_output, axis=0) / numAmostras (soma os deltas ao longo das amostras para cada neurônio de saída)
        // O resultado db2 é um vetor linha [1, tamanhoSaida], igual a b2.
        this.gradB2 = new SimpleMatrix(1, this.parametrosRede.tamanhoSaida());
        for (int j = 0; j < deltaSaida.getNumCols(); j++) { // Itera sobre neurônios de saída
            double somaColuna = 0;
            for (int i = 0; i < deltaSaida.getNumRows(); i++) { // Itera sobre amostras
                somaColuna += deltaSaida.get(i, j);
            }
            this.gradB2.set(0, j, somaColuna / numAmostras); // Média do gradiente do bias
        }


        // --- Etapa 2: Calcular Gradientes para a Camada Escondida (W1, b1) ---

        // Propagar o delta da saída para a camada escondida
        // erro_escondido = delta_output @ W2^T
        // dE/da1 = dE/dz2 * dz2/da1
        // dz2/da1 é W2, então dE/da1 = delta_output @ W2.T
        // hiddenError tem shape [numAmostras, tamanhoCamadaEscondida]
        SimpleMatrix erroCamadaEscondida = deltaSaida.mult(this.W2.transpose());

        // Derivada da função de ativação Sigmoid na camada escondida (calculada em z1, usando a1)
        // da1/dz1 = a1 * (1 - a1)
        SimpleMatrix derivadaSigmoidEscondida = derivadaSigmoid(this.a1);

        // Delta da camada escondida (delta_j ou delta_hidden)
        // delta_hidden = (erro propagado para a camada escondida) * (derivada da ativação escondida) --- multiplicação elemento a elemento
        // delta_hidden = dE/da1 * da1/dz1 = dE/dz1
        SimpleMatrix deltaEscondido = erroCamadaEscondida.elementMult(derivadaSigmoidEscondida); // Shape: [numAmostras, tamanhoCamadaEscondida]

        // Gradiente para os pesos W1 (dE/dW1)
        // dE/dW1 = (X^T @ delta_hidden) / numAmostras
        // Onde X é a entrada da rede (armazenada em this.ultimaEntradaX)
        // X^T tem shape [tamanhoEntrada, numAmostras]
        // delta_hidden tem shape [numAmostras, tamanhoCamadaEscondida]
        // O resultado dW1 tem shape [tamanhoEntrada, tamanhoCamadaEscondida], igual a W1.
        this.gradW1 = this.ultimaEntradaX.transpose().mult(deltaEscondido).divide(numAmostras);

        // Gradiente para o bias b1 (dE/db1)
        // dE/db1 = sum(delta_hidden, axis=0) / numAmostras
        // O resultado db1 é um vetor linha [1, tamanhoCamadaEscondida], igual a b1.
        this.gradB1 = new SimpleMatrix(1, this.parametrosRede.tamanhoCamadaEscondida());
        for (int j = 0; j < deltaEscondido.getNumCols(); j++) { // Itera sobre neurônios escondidos
            double somaColuna = 0;
            for (int i = 0; i < deltaEscondido.getNumRows(); i++) { // Itera sobre amostras
                somaColuna += deltaEscondido.get(i, j);
            }
            this.gradB1.set(0, j, somaColuna / numAmostras); // Média do gradiente do bias
        }
    }

    /**
     * Atualiza os pesos e biases da rede usando os gradientes calculados
     * pelo método `backward` e a taxa de aprendizado fornecida.
     * Implementa a regra de atualização do Gradiente Descendente:
     * novo_peso = peso_antigo - taxa_aprendizado * gradiente_peso
     * novo_bias = bias_antigo - taxa_aprendizado * gradiente_bias
     *
     * @param taxaAprendizado O fator que controla o tamanho do passo de atualização.
     * @throws IllegalStateException Se os gradientes ainda não foram calculados (backward() não foi chamado).
     */
    public void atualizarPesos(double taxaAprendizado) {
        // Verifica se os gradientes foram calculados antes de tentar usá-los
        if (this.gradW1 == null || this.gradB1 == null || this.gradW2 == null || this.gradB2 == null) {
            throw new IllegalStateException("Execute backward() antes de atualizarPesos() para calcular os gradientes.");
        }

        // Atualização W1 = W1 - learningRate * dW1
        // .scale() multiplica a matriz pelo escalar
        // .minus() subtrai matrizes elemento a elemento
        this.W1 = this.W1.minus(this.gradW1.scale(taxaAprendizado));

        // Atualização b1 = b1 - learningRate * db1
        this.b1 = this.b1.minus(this.gradB1.scale(taxaAprendizado));

        // Atualização W2 = W2 - learningRate * dW2
        this.W2 = this.W2.minus(this.gradW2.scale(taxaAprendizado));

        // Atualização b2 = b2 - learningRate * db2
        this.b2 = this.b2.minus(this.gradB2.scale(taxaAprendizado));
    }

    /**
     * Executa o ciclo completo de treinamento, opcionalmente usando parada antecipada
     * baseada em um conjunto de validação.
     *
     * @param xTreino Matriz de dados de entrada para treinamento.
     * @param yTreino Matriz de rótulos verdadeiros (one-hot) para treinamento.
     * @param xValidacao Matriz de dados de entrada para validação (pode ser null se não usar parada antecipada).
     * @param yValidacao Matriz de rótulos verdadeiros (one-hot) para validação (pode ser null).
     * @param parametrosTreinamento Definicao dos hiperparametros para o treinamento.
     * @return A época em que o treinamento parou (seja pelo maxEpocas ou parada antecipada).
     */
    public int treinar(SimpleMatrix xTreino, SimpleMatrix yTreino,
                       SimpleMatrix xValidacao, SimpleMatrix yValidacao,
                       ParametrosTreinamento parametrosTreinamento) {
        this.historicoErro.clear();
        this.historicoErroValidacao.clear();
        this.melhorErroValidacao = Double.MAX_VALUE;
        this.epocaMelhorErro = 0;
        int contadorPaciencia = 0;
        boolean usarParadaAntecipada = (xValidacao != null && yValidacao != null && parametrosTreinamento.pacienciaParadaAntecipada() > 0);

        System.out.println("Iniciando treinamento por no maximo " + parametrosTreinamento.epocas() + " epocas...");
        if (usarParadaAntecipada) {
            System.out.println("Parada Antecipada ativada com paciência de " + parametrosTreinamento.pacienciaParadaAntecipada() + " epocas.");
        }

        for (int epoca = 0; epoca < parametrosTreinamento.epocas(); epoca++) {
            // 1. Passo Forward (Treino)
            SimpleMatrix yPrevistoTreino = this.forward(xTreino);

            // 2. Cálculo do Erro de Treino
            double erroTreinoAtual = this.erroQuadraticoMedio(yTreino, yPrevistoTreino);
            this.historicoErro.add(erroTreinoAtual);

            // 3. Passo Backward
            this.backward(yTreino);

            // 4. Atualização de Pesos
            this.atualizarPesos(parametrosTreinamento.taxaAprendizado());

            // 5. Avaliação no Conjunto de Validação e Parada Antecipada (se ativada)
            double erroValidacaoAtual = Double.NaN; // Valor padrão se não houver validação
            if (usarParadaAntecipada) {
                // Calcula a saída para o conjunto de validação (SEM armazenar estados intermediários do forward)
                SimpleMatrix yPrevistoValidacao = this.forward(xValidacao);
                erroValidacaoAtual = this.erroQuadraticoMedio(yValidacao, yPrevistoValidacao);
                this.historicoErroValidacao.add(erroValidacaoAtual);

                // Lógica da Parada Antecipada
                if (erroValidacaoAtual < this.melhorErroValidacao) {
                    this.melhorErroValidacao = erroValidacaoAtual;
                    this.epocaMelhorErro = epoca + 1; // Armazena a época atual (base 1)
                    contadorPaciencia = 0; // Reseta a paciência
                    // Salva os pesos atuais como os melhores
                    this.melhorW1 = this.W1.copy();
                    this.melhorB1 = this.b1.copy();
                    this.melhorW2 = this.W2.copy();
                    this.melhorB2 = this.b2.copy();
                } else {
                    contadorPaciencia++; // Incrementa a paciência
                }

                // Verifica se a paciência estourou
                if (contadorPaciencia >= parametrosTreinamento.pacienciaParadaAntecipada()) {
                    System.out.println("Parada Antecipada acionada na epoca " + (epoca + 1) +
                            ". Melhor erro de validação (" + String.format("%.8f", this.melhorErroValidacao) +
                            ") ocorreu na epoca " + this.epocaMelhorErro + ".");
                    // Restaura os melhores pesos encontrados
                    restaurarMelhoresPesos();
                    return epoca + 1; // Retorna a época em que parou
                }
            }

            // Imprimir progresso
            if ((epoca + 1) % 1000 == 0 || epoca == 0) {
                if (usarParadaAntecipada) {
                    System.out.printf("Epoca %d/%d, Erro Treino: %.8f, Erro Validacaoo: %.8f (Melhor: %.8f na epoca %d)\n",
                            epoca + 1, parametrosTreinamento.epocas(), erroTreinoAtual, erroValidacaoAtual, melhorErroValidacao, epocaMelhorErro);
                } else {
                    System.out.printf("Epoca %d/%d, Erro Treino: %.8f\n", epoca + 1, parametrosTreinamento.epocas(), erroTreinoAtual);
                }
            }
        }

        System.out.println("Treinamento concluido (atingiu maximo de Epocas).");
        // Se a parada antecipada estava ativa e nunca foi acionada,
        // restaura os melhores pesos (que podem ser os da última época ou anteriores).
        if (usarParadaAntecipada && this.melhorW1 != null) {
            restaurarMelhoresPesos();
        }

        return parametrosTreinamento.epocas();
    }

    /**
     * Realiza previsões para um novo conjunto de dados de entrada usando a rede treinada.
     * Essencialmente, executa apenas o passo forward.
     *
     * @param xTeste Matriz com as amostras de entrada para as quais fazer a previsão.
     * @return A matriz de saída da rede (probabilidades ou ativações) para as entradas de teste.
     */
    public SimpleMatrix prever(SimpleMatrix xTeste) {
        return this.forward(xTeste);
    }

    /**
     * Restaura os pesos e biases da rede para os melhores valores salvos
     * durante o treinamento com parada antecipada.
     */
    private void restaurarMelhoresPesos() {
        if (this.melhorW1 != null) { // Verifica se houve pelo menos uma melhoria
            this.W1 = this.melhorW1.copy();
            this.b1 = this.melhorB1.copy();
            this.W2 = this.melhorW2.copy();
            this.b2 = this.melhorB2.copy();
        } else {
            System.out.println("Aviso: Nenhum peso melhor foi salvo durante a validação (erro de validação nunca melhorou?). Usando pesos da última época.");
        }
    }
}