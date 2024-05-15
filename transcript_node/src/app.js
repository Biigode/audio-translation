import { HfInference } from "@huggingface/inference";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import fs, { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import pkg from "wavefile";
const { WaveFile } = pkg;

const HF_TOKEN = process.env.HF_TOKEN;

const hf = new HfInference(HF_TOKEN);

const readJsonFile = (fileName) => {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const outputFilePath = path.join(__dirname, `transcribe_audios/${fileName}`);
  const jsonData = fs.readFileSync(outputFilePath, "utf-8");
  const parsedData = JSON.parse(jsonData);
  return parsedData;
};

const hfEmbeddings = async () => {
  /* gerar embbeding */
  const model = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });

  /* Embed queries */
  const res = await model.embedQuery(
    "What would be a good company name for a company that makes colorful socks?"
  );
  console.log({ res });
  /* Embed documents */
  const documentRes = await model.embedDocuments(["Hello world", "Bye bye"]);
  console.log({ documentRes });
};

// await hfEmbeddings();

const hfQuestionsAnswring = async () => {
  const parsedData = readJsonFile("output.json");
  const answered = await hf.questionAnswering({
    model: "deepset/roberta-base-squad2",
    inputs: {
      question: "Sobre o que se trata esse contexto?",
      context: parsedData.text,
    },
  });

  console.log({ answered });

  // const parsedData = readJsonFile("output.json");
  // const answerer = await pipeline(
  //   "question-answering",
  //   "Xenova/distilbert-base-uncased-distilled-squad"
  // );
  // const question =
  //   "Quais são as possiveis perguntas para que posso fazer dado esse contexto?";
  // // const context = "My dog's name is Max and he is a black labrador.";
  // const output = await answerer(question, parsedData.text);
  // console.log({ output });
};

const hfTransformAudioToText = async () => {
  // const transcriber = await pipeline(
  //   "automatic-speech-recognition",
  //   "Xenova/whisper-largev3"
  // );
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const audioPath = path.join(__dirname, "audio/Gravando.mp3");

  const file_buffer = readFileSync(audioPath);

  // console.log({ file_buffer });

  const audioReaded = await hf.automaticSpeechRecognition({
    model: "openai/whisper-large-v3",
    data: file_buffer,
  });

  console.log({ audioReaded });

  // // Carregar dados de áudio
  // let buffer = fs.readFileSync(audioPath);

  // // Ler o arquivo .wav e convertê-lo para o formato necessário

  // let wav = new WaveFile(file_buffer);
  // wav.toBitDepth("32f"); // A pipeline espera a entrada como um Float32Array
  // wav.toSampleRate(16000); // Whisper espera áudio com uma taxa de amostragem de 16000
  // let audioData = wav.getSamples();
  // if (Array.isArray(audioData)) {
  //   if (audioData.length > 1) {
  //     const SCALING_FACTOR = Math.sqrt(2);

  //     // Mesclar canais (no primeiro canal para economizar memória)
  //     for (let i = 0; i < audioData[0].length; ++i) {
  //       audioData[0][i] =
  //         (SCALING_FACTOR * (audioData[0][i] + audioData[1][i])) / 2;
  //     }
  //   }

  //   // Selecionar o primeiro canal
  //   audioData = audioData[0];
  // }

  // const output = await transcriber(audioData, { return_timestamps: true });
  // console.log(JSON.stringify({ output }));
  // const outputFilePath = path.join(__dirname, "transcribe_audios/output.json");
  // fs.writeFileSync(outputFilePath, JSON.stringify(output));
  // console.log("Output saved to:", outputFilePath);
};

const hfDataSumarization = async () => {
  const parsedData = readJsonFile("output.json");
  const sumarization = await hf.summarization({
    model: "facebook/bart-large-cnn",
    inputs: parsedData.text,
    parameters: {
      max_length: 300,
    },
  });
  console.log({ sumarization });

  // const generator = await pipeline("summarization", "Xenova/bart-large-cnn");
  // const output = await generator(parsedData.text, {
  //   max_new_tokens: 100,
  // });
  // console.log({ output });
};

const text2Text = async () => {
  // const parsedData = readJsonFile("output.json");
  // const generator = await pipeline(
  //   "text2text-generation",
  //   "Xenova/long-t5-tglobal-base"
  // );
  // const output = await generator(
  //   [
  //     `question: Quais perguntas posso fazer dado esse contexto? context: ${parsedData.text}`,
  //   ],
  //   {
  //     max_new_tokens: 100,
  //   }
  // );
  // console.log({ output });
};

// await hfTransformAudioToText();
// await hfDataSumarization();
await hfQuestionsAnswring();
// await text2Text();
