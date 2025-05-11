## Pré-requisitos
*   JDK 22 (ou compatível) instalado e configurado (com `JAVA_HOME` definido).
*   Gradle instalado OU acesso ao Gradle Wrapper (`gradlew` ou `gradlew.bat`) na raiz do projeto.

## Estrutura de Dados
Os arquivos de dataset `X.txt` e `Y_letra.txt` devem estar localizados em `src/main/resources/datasets/caracteres/`.

## Compilando e Executando via Linha de Comando

Navegue até o diretório raiz do projeto no seu terminal.

**1. Compilar o Projeto:**
Use o Gradle Wrapper para compilar o código e criar os artefatos necessários:

*   No Linux/macOS:
    ```bash
    ./gradlew build
    ```
*   No Windows:
    ```bash
    .\gradlew.bat build
    ```
Isso irá compilar as classes e pode executar testes se estiverem configurados.

**2. Executar a Aplicação Principal:**
Após a compilação bem-sucedida, você pode executar a aplicação usando a tarefa `run` do Gradle, que executará o método `main` definido no `build.gradle.kts` (provavelmente `br.com.usp.ach2016.Main`):

*   No Linux/macOS:
    ```bash
    ./gradlew run
    ```
*   No Windows:
    ```bash
    .\gradlew.bat run
    ```
