package br.com.usp.ach2016.executions;

import br.com.usp.ach2016.model.Dataset;
import br.com.usp.ach2016.model.ParametrosRede;
import br.com.usp.ach2016.model.ParametrosTreinamento;

import static br.com.usp.ach2016.utils.IOUtils.criarPastaResultadosNExecucao;

public abstract class Execucao {

    final Dataset dataset;
    ParametrosRede parametrosRede;
    final ParametrosTreinamento parametrosTreinamento;
    final String caminhoPastaResultadosExecucao;

     Execucao(Dataset dataset,
              ParametrosRede parametrosRede,
              ParametrosTreinamento parametrosTreinamento) {
        this.dataset = dataset;
        this.parametrosRede = parametrosRede;
        this.parametrosTreinamento = parametrosTreinamento;
        this.caminhoPastaResultadosExecucao = criarPastaResultadosNExecucao(parametrosRede.nomeExecucao());
    }

    public abstract void executar();
}
