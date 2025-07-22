# Consulta Inteligente de Documentos

Este projeto apresenta um **pipeline completo**, projetado para capacitar modelos de linguagem a responderem perguntas com base em um conjunto de documentos heterogêneos. A solução demonstra a capacidade de integrar diversas tecnologias de IA para extrair, indexar e recuperar informações de forma inteligente.

## Objetivo do Projeto

O objetivo principal deste desafio foi desenvolver uma solução funcional que:

* **Indexe documentos heterogêneos:** Suporte para PDFs com texto nativo, PDFs digitalizados (imagens de documentos) e imagens (JPEG/PNG/WEBP contendo texto).
* **Aplique OCR (Optical Character Recognition) automaticamente:** Identifica a necessidade de OCR e o aplica de forma inteligente para extrair texto de conteúdos não selecionáveis.
* **Gere uma base vetorial:** Converte o texto extraído em representações numéricas (embeddings) e as armazena de forma eficiente.
* **Implemente recuperação de informações:** Busca os trechos mais relevantes do seu banco de dados vetorial em resposta a uma pergunta em linguagem natural.
* **Utilize uma Large Language Model (LLM) para responder perguntas:** Combina os trechos recuperados com uma LLM para gerar respostas contextualmente relevantes e precisas.

## Arquitetura do Pipeline

O pipeline é modular e segue a abordagem RAG, dividindo o processo em três etapas principais para garantir eficiência e escalabilidade:

1.  ### **Extração de Conteúdo e Pré-processamento**
    Nesta fase, o sistema é capaz de ler e processar diversos formatos de documentos:
    * **PDFs com texto nativo:** Utiliza `pdfplumber` para uma extração de texto precisa, mantendo a estrutura original (incluindo tabelas).
    * **PDFs digitalizados e Imagens (JPEG, PNG, WEBP):** Quando o texto não é nativamente selecionável (em PDFs digitalizados) ou está em formato de imagem, o **OCR** é acionado. `pytesseract` (interface para o **Tesseract OCR**) é empregado para converter pixels em texto.
    * **Pré-processamento de Imagens:** Antes de alimentar as imagens ao Tesseract, `opencv-python-headless` é utilizado para aplicar técnicas de pré-processamento como conversão para escala de cinza, binarização (com o método de Otsu para ajuste automático de limiar) e redimensionamento. Essas etapas otimizam a qualidade da imagem para o OCR, **aumentando significativamente a acurácia da extração de texto**, especialmente em documentos com variações de qualidade.

2.  ### **Indexação Semântica**
    Após a extração do texto, as informações são preparadas para uma recuperação eficiente:
    * **Chunking:** O texto completo de todos os documentos é dividido em **blocos (`chunks`) de 800 palavras**. Essa estratégia garante que cada pedaço de informação seja grande o suficiente para conter contexto, mas pequeno o suficiente para ser processado pela LLM.
    * **Geração de Embeddings:** Cada chunk é transformado em um **vetor semântico (embedding)** usando o modelo `SentenceTransformer` (`all-MiniLM-L6-v2`). Esses vetores capturam o significado contextual do texto, permitindo comparações semânticas.
    * **Armazenamento Vetorial:** Os embeddings são armazenados e indexados em um banco de dados **FAISS (Facebook AI Similarity Search)**. O FAISS é escolhido por sua alta performance em buscas de similaridade de vizinhos mais próximos em espaços de alta dimensão, essencial para a recuperação rápida de informações relevantes.

3.  ### **Recuperação e Geração com LLM**
    Esta é a fase final, onde o sistema interage com o usuário e gera respostas:
    * **Pergunta em Linguagem Natural:** O usuário insere uma pergunta em linguagem natural.
    * **Busca de Similaridade:** A pergunta do usuário é convertida em um embedding (usando o mesmo `SentenceTransformer`) e utilizada para consultar o índice FAISS. Os **trechos (`chunks`) mais semanticamente relevantes** são recuperados.
    * **Geração de Resposta pela LLM:** Os trechos recuperados, juntamente com a pergunta original, são enviados ao modelo **`gemini-2.5-flash`** da Google via `google-generativeai` API. A LLM utiliza esse contexto fornecido para gerar uma resposta concisa, relevante e baseada nas informações contidas nos documentos. O prompt é construído para instruir a LLM a **responder estritamente com base no contexto fornecido**, minimizando alucinações.

## Tecnologias Utilizadas

* **`Python`**: A linguagem de escolha pela sua maturidade e ecossistema robusto para IA, oferecendo bibliotecas de ponta para PNL, visão computacional e aprendizado de máquina. Sua sintaxe limpa e legibilidade aceleram a prototipação e o desenvolvimento.

* **`pdfplumber`**: Uma biblioteca Python para extração de texto de PDFs. Foi selecionada por sua capacidade de lidar com PDFs com texto nativo de forma eficaz, extraindo não apenas o texto, mas também informações de layout como tabelas e coordenadas, que poderiam ser úteis em desenvolvimentos futuros.

* **`pytesseract` (com Tesseract OCR)**: Um wrapper Python para o popular motor OCR de código aberto Tesseract. Escolhido pela sua robustez e flexibilidade, incluindo suporte nativo a múltiplos idiomas (configurado para `português`) e a capacidade de ser acionado condicionalmente para PDFs digitalizados e imagens.

* **`opencv-python-headless`**: A biblioteca OpenCV é muito utilizada para o pré-processamento de imagens. A versão `headless` é ideal para ambientes de servidor/Colab, pois não requer interface gráfica. As operações de binarização (Otsu) e redimensionamento são cruciais para otimizar a entrada para o Tesseract, melhorando a acurácia do OCR em imagens variadas.

* **`sentence-transformers`**: Uma biblioteca para gerar embeddings de sentenças, parágrafos e documentos. O modelo `all-MiniLM-L6-v2` foi escolhido por seu equilíbrio entre desempenho, velocidade e tamanho compacto, tornando-o eficiente para gerar embeddings de alta qualidade sem exigir grandes recursos computacionais.

* **`faiss-cpu`**: Desenvolvido pelo Facebook AI, o FAISS é uma biblioteca para pesquisa eficiente de similaridade e agrupamento de vetores de alta dimensão. A versão `faiss-cpu` foi escolhida para execução local (ou em ambientes como Colab), fornecendo um desempenho de busca de vizinhos mais próximos extremamente rápido e escalável para milhões de vetores.

* **`google-generativeai` (Gemini API)**: A interface Python para os modelos Gemini da Google. O `gemini-2.5-flash` é um modelo de linguagem grande otimizado para tarefas de resposta rápida e eficiência, ideal para ser a LLM no pipeline RAG, processando o contexto recuperado e gerando respostas coerentes e informativas.


## Guia de Execução (via Google Colab)

Siga os passos abaixo para configurar e executar o pipeline no Google Colab:

1.  ### **Acesse o Notebook:**
    Clique no link para abrir o notebook no Google Colab:
    [📎 Desafio.ipynb](https://colab.research.google.com/drive/1BPBTaotNlB6AL570uBKyZ9p_xvhemw2A?usp=sharing)

2.  ### **Faça Upload dos Documentos:**
    Execute a célula que utiliza `from google.colab import files; uploaded = files.upload()`. Uma caixa de diálogo aparecerá, permitindo que você selecione os arquivos do seu computador. Faça upload dos documentos fornecidos para o desafio:
    * `CÓDIGO DE OBRAS.pdf`
    * `tabela.webp`

3.  ### **Execute as Etapas do Pipeline (Célula por Célula):**
    Prossiga executando as células do notebook sequencialmente:
    * **Etapa 1: Extração de Texto com OCR Condicional:** Esta célula processará os documentos enviados, extraindo texto nativo ou aplicando OCR conforme necessário. O texto consolidado será salvo em `extracted_text.txt`.
    * **Etapa 2: Chunking, Embeddings e Indexação:** Esta célula lerá o `extracted_text.txt`, dividirá o conteúdo em chunks, gerará embeddings para cada chunk e os indexará no FAISS.
    * **Etapa 3: Perguntas e Respostas via Gemini:** Esta é a etapa interativa. Após a execução, você será solicitado a digitar suas perguntas no console. O sistema recuperará o contexto relevante e usará o Gemini para gerar respostas. Digite 'sair' para encerrar a sessão.


## Autor
Desenvolvido por Melissa Guedes.
