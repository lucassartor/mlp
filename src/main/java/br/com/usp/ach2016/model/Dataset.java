package br.com.usp.ach2016.model;

import org.ejml.simple.SimpleMatrix;

import java.util.List;
import java.util.Map;

/**
 * Estrutura de dados (record) para armazenar o conjunto de dados de caracteres dividido.
 * Inclui matrizes de treino/teste e informações sobre as classes.
 *
 * @param xTreino Matriz com os dados de entrada (pixels) para treinamento. Shape: [numAmostrasTreino, numAtributos].
 * @param yTreino Matriz com os rótulos (one-hot encoded) para treinamento. Shape: [numAmostrasTreino, numClasses].
 * @param xTeste Matriz com os dados de entrada (pixels) para teste. Shape: [numAmostrasTeste, numAtributos].
 * @param yTeste Matriz com os rótulos (one-hot encoded) para teste. Shape: [numAmostrasTeste, numClasses].
 * @param rotulosClasses Lista ordenada dos rótulos de classe únicos (ex: ['A', 'B', ..., 'Z']).
 * @param caractereParaIndice Mapa que associa cada caractere ao seu índice correspondente na lista `rotulosClasses` e na codificação one-hot.
 */
public record Dataset(
        SimpleMatrix xTreino, SimpleMatrix yTreino,
        SimpleMatrix xValidacao, SimpleMatrix yValidacao,
        SimpleMatrix xTeste, SimpleMatrix yTeste,
        List<Character> rotulosClasses,
        Map<Character, Integer> caractereParaIndice
) {}