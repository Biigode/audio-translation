import { HfInference, chatCompletion } from "@huggingface/inference";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import fs, { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

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

(async function init() {
  const hfTransformAudioToText = async () => {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const audioPath = path.join(__dirname, "audio/Gravando2.mp3");

    const file_buffer = readFileSync(audioPath);

    const audioReaded = await hf.automaticSpeechRecognition({
      model: "openai/whisper-large-v3",
      data: file_buffer,
    });

    return audioReaded;
  };

  const textTranslation = async (srcLanguage, targetLanguage, input) => {
    const translatedText = await hf.translation({
      model: "facebook/mbart-large-50-many-to-many-mmt",
      inputs: input,
      parameters: {
        src_lang: srcLanguage,
        tgt_lang: targetLanguage,
      },
    });

    return translatedText;
  };

  const hfDataSumarization = async (input) => {
    const sumarization = await hf.summarization({
      model: "facebook/bart-large-cnn",
      inputs: input,
      parameters: {
        max_length: 300,
        temperature: 0.1,
      },
    });
    return sumarization;
  };

  //TODO: Validar se existe algo que possa ser feito com isso
  const hfTextGeneration = async (input) => {
    const generatedText = await hf.textGeneration({
      model: "google/gemma-7b",
      inputs: `Context: ${input}.
      Question: What are the possible questions I can ask given this context?
      Make a summary of the text.
      Provide the answer for each question.`,
      parameters: {
        max_new_tokens: 1250,
        temperature: 0.1,
      },
    });

    return generatedText;
  };

  const hfConversational = async (input, question) => {
    const conversation = await chatCompletion({
      model: "google/gemma-1.1-7b-it",
      accessToken: HF_TOKEN,
      messages: [
        {
          role: "user",
          content: `Using this context: ${input}.
            ${question}`,
        },
      ],
      max_tokens: 500,
      temperature: 0.1,
      seed: 0,
    });

    return conversation;
  };

  try {
    const audioTranslated = await hfTransformAudioToText();
    console.log(audioTranslated.text);
    const translatedText = await textTranslation(
      "pt_XX",
      "en_XX",
      audioTranslated.text
    );
    console.log(translatedText.translation_text);
    const sumarization = await hfDataSumarization(
      translatedText.translation_text
    );
    console.log(sumarization.summary_text);
    const firstQuestion =
      "What are the possible questions I can ask given this context?";
    const firstChatResponse = await hfConversational(
      sumarization.summary_text,
      firstQuestion
    );
    console.log("############################################");
    console.log(firstChatResponse.choices[0].message.content);

    const secondQuestion =
      "Based on the preview context, answser all the question.";
    const secondChatResponse = await hfConversational(
      firstChatResponse.choices[0].message.content,
      secondQuestion
    );
    console.log("############################################");
    console.log(secondChatResponse.choices[0].message.content);
  } catch (error) {
    console.log(error);
  }
})();
