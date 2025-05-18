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
import static br.com.usp.ach2016.utils.MatrixUtils.*;
import static br.com.usp.ach2016.utils.MetricsUtils.formatarDuracao;

public class ExecucaoSimples extends Execucao {

    public ExecucaoSimples(Dataset dataset,
                           ParametrosRede parametrosRede,
                           ParametrosTreinamento parametrosTreinamento) {
        super(dataset, parametrosRede, parametrosTreinamento);
    }

    @Override
    public void executar() {
        String nomeExecucao = criarPastaResultadosExecucao(parametrosRede.nomeExecucao());

        this.parametrosRede = new ParametrosRede(
                parametrosRede.tamanhoEntrada(),
                parametrosRede.tamanhoCamadaEscondida(),
                parametrosRede.tamanhoSaida(),
                parametrosRede.sementeAleatoria(),
                nomeExecucao
        );

        System.out.println("\n\n===== INICIANDO: " + parametrosRede.nomeExecucao().toUpperCase() + " =====");


        // Juntando conjunto de treino + de validação (totalizando 1196 exemplos para TREINO)
        SimpleMatrix xTreinoCompleto = combinarMatrizesVerticalmente(dataset.xTreino(), dataset.xValidacao());
        SimpleMatrix yTreinoCompleto = combinarMatrizesVerticalmente(dataset.yTreino(), dataset.yValidacao());

        MLP redeNeural = new MLP(parametrosRede);
        salvarPesosIniciais(caminhoPastaResultadosExecucao,
                           redeNeural.pesosIniciaisW1,
                           redeNeural.biasInicialB1,
                           redeNeural.pesosIniciaisW2,
                           redeNeural.biasInicialB2);

        long inicioTreino = System.currentTimeMillis();
        redeNeural.treinar(xTreinoCompleto, yTreinoCompleto, null, null, parametrosTreinamento);
        String duracaoTreinoFormatada = formatarDuracao(System.currentTimeMillis() - inicioTreino);
        System.out.println("Tempo de Treinamento: " + duracaoTreinoFormatada);
        salvarResultadosTreino(caminhoPastaResultadosExecucao, parametrosTreinamento, redeNeural);

        SimpleMatrix previsoesTeste = redeNeural.prever(dataset.xTeste());

        int[][] matrizConfusao = calcularMatrizConfusao(dataset.yTeste(), previsoesTeste);
        salvarMatrizConfusao(caminhoPastaResultadosExecucao, matrizConfusao, dataset.rotulosClasses());

        Map<String, Object> paramsAdicionaisTreino = new HashMap<>();
        paramsAdicionaisTreino.put("Tempo de Treinamento", duracaoTreinoFormatada); // Adiciona a duração formatada

        //salvarHistoricoDeltaPesos(caminhoPastaResultadosExecucao, redeNeural.historicoMediaDeltaW1, "W1");
        //salvarHistoricoDeltaPesos(caminhoPastaResultadosExecucao, redeNeural.historicoMediaDeltaW2, "W2");

        ResultadoAnaliseConfusao resultadoAnaliseConfusao = gerarResultadosClassificacao(matrizConfusao, dataset, previsoesTeste);
        salvarResultadosTeste(caminhoPastaResultadosExecucao,
                              dataset,
                              previsoesTeste,
                              parametrosRede,
                              parametrosTreinamento,
                              paramsAdicionaisTreino,
                              resultadoAnaliseConfusao
        );

        System.out.println("===== FINALIZADA: " + parametrosRede.nomeExecucao().toUpperCase() + " =====");
    }

}