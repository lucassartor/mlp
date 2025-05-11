package br.com.usp.ach2016.executions;

import br.com.usp.ach2016.MLP;
import br.com.usp.ach2016.model.Dataset;
import br.com.usp.ach2016.model.ParametrosRede;
import br.com.usp.ach2016.model.ParametrosTreinamento;
import br.com.usp.ach2016.model.ResultadoAnaliseConfusao;
import org.ejml.simple.SimpleMatrix;

import static br.com.usp.ach2016.utils.IOUtils.*;
import static br.com.usp.ach2016.utils.MatrixUtils.calcularMatrizConfusao;
import static br.com.usp.ach2016.utils.MatrixUtils.gerarResultadosClassificacao;

public class ExecucaoSimples extends Execucao {

    public ExecucaoSimples(Dataset dataset,
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

        redeNeural.treinar(dataset.xTreino(), dataset.yTreino(), null, null, parametrosTreinamento);
        salvarResultadosTreino(caminhoPastaResultadosExecucao, parametrosTreinamento, redeNeural);

        SimpleMatrix previsoesTeste = redeNeural.prever(dataset.xTeste());

        int[][] matrizConfusao = calcularMatrizConfusao(dataset.yTeste(), previsoesTeste);
        salvarMatrizConfusao(caminhoPastaResultadosExecucao, matrizConfusao, dataset.rotulosClasses());

        ResultadoAnaliseConfusao resultadoAnaliseConfusao = gerarResultadosClassificacao(matrizConfusao, dataset, previsoesTeste);
        salvarResultadosTeste(caminhoPastaResultadosExecucao, dataset, previsoesTeste, parametrosRede, parametrosTreinamento, null, resultadoAnaliseConfusao);

        System.out.println("===== FINALIZADA: " + parametrosRede.nomeExecucao().toUpperCase() + " =====");
    }

}