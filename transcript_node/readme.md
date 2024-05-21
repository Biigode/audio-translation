

# Hugging Face API Integration

Este projeto demonstra como utilizar as APIs do Hugging Face para diversas tarefas, incluindo reconhecimento de fala, tradução de textos, sumarização, geração de texto e embeddings.

## Pré-requisitos

- Node.js instalado
- Uma conta no Hugging Face com um token de acesso (HF_TOKEN)
https://huggingface.co/settings/tokens

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

2. Instale as dependências:

```bash
npm install
```

3. Configure o arquivo `.env`:

Crie um arquivo `.env` na raiz do projeto e adicione seu token do Hugging Face:

```env
HF_TOKEN=seu_token_do_hugging_face
```

## Uso

Para executar o script, utilize o comando:

```bash
node --env-file=.env src/app.js
```

## Descrição

Este script realiza as seguintes tarefas utilizando as APIs do Hugging Face:

1. **Reconhecimento de Fala:** Converte áudio em texto utilizando o modelo `openai/whisper-large-v3`.
2. **Tradução de Texto:** Traduz o texto reconhecido para outro idioma utilizando o modelo `facebook/mbart-large-50-many-to-many-mmt`.
3. **Sumarização:** Gera um resumo do texto traduzido utilizando o modelo `facebook/bart-large-cnn`.
4. **Geração de Texto:** Gera texto baseado em um contexto fornecido utilizando o modelo `google/gemma-7b`.
5. **Conversação:** Realiza uma conversa baseada em um contexto e perguntas fornecidas utilizando o modelo `google/gemma-1.1-7b-it`.
6. **Embeddings:** Gera embeddings para consultas e documentos utilizando o modelo `Xenova/all-MiniLM-L6-v2`.

---

Espero que isso ajude! Se precisar de mais alguma coisa, estou à disposição.
