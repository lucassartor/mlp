package br.com.usp.ach2016.executions;

import br.com.usp.ach2016.MLP;
import br.com.usp.ach2016.model.Dataset;
import br.com.usp.ach2016.model.ParametrosRede;
import br.com.usp.ach2016.model.ParametrosTreinamento;
import br.com.usp.ach2016.model.ResultadoAnaliseConfusao;
import org.ejml.simple.SimpleMatrix;

import java.util.HashMap;
import java.util.Map;

import static br.com.usp.ach2016.utils.IOUtils.*;
import static br.com.usp.ach2016.utils.MatrixUtils.calcularMatrizConfusao;
import static br.com.usp.ach2016.utils.MatrixUtils.gerarResultadosClassificacao;

public class ExecucaoComParadaAntecipada extends Execucao {

    public ExecucaoComParadaAntecipada(Dataset dataset,
                                       ParametrosRede parametrosRede,
                                       ParametrosTreinamento parametrosTreinamento) {
        super(dataset, parametrosRede, parametrosTreinamento);
    }

    @Override
    public void executar() {
        System.out.println("\n\n===== INICIANDO: " + parametrosRede.nomeExecucao().toUpperCase() + " =====");

        MLP redeNeural = new MLP(parametrosRede);
        salvarPesosIniciais(caminhoPastaResultadosExecucao,
                            redeNeural.pesosIniciaisW1,
                            redeNeural.biasInicialB1,
                            redeNeural.pesosIniciaisW2,
                            redeNeural.biasInicialB2);

        int epocasExecutadas = redeNeural.treinar(dataset.xTreino(), dataset.yTreino(),  dataset.xValidacao(), dataset.yValidacao(), parametrosTreinamento);
        salvarResultadosTreino(caminhoPastaResultadosExecucao, parametrosTreinamento, redeNeural);

        System.out.println("Treinamento com parada antecipada finalizado na epoca: " + epocasExecutadas);

        SimpleMatrix previsoesTeste = redeNeural.prever(dataset.xTeste());

        int[][] matrizConfusao = calcularMatrizConfusao(dataset.yTeste(), previsoesTeste);
        salvarMatrizConfusao(caminhoPastaResultadosExecucao, matrizConfusao, dataset.rotulosClasses());

        ResultadoAnaliseConfusao resultadoAnaliseConfusao = gerarResultadosClassificacao(matrizConfusao, dataset, previsoesTeste);
        Map<String, Object> paramsAdicionaisTreino = new HashMap<>();
        paramsAdicionaisTreino.put("Épocas Efetivas", epocasExecutadas);
        paramsAdicionaisTreino.put("Melhor Época (Validação)", redeNeural.epocaMelhorErro);
        paramsAdicionaisTreino.put("Melhor Erro Validação", String.format("%.8f", redeNeural.melhorErroValidacao));

        salvarResultadosTeste(caminhoPastaResultadosExecucao, dataset, previsoesTeste, parametrosRede, parametrosTreinamento, paramsAdicionaisTreino, resultadoAnaliseConfusao);

        System.out.println("===== FINALIZADA: " + parametrosRede.nomeExecucao().toUpperCase() + " =====");
    }
}
