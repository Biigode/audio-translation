import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from datasets import load_dataset
import torch
import os


class Transcript:

    def __init__(self):
   
        self.torch_dtype = torch.float32
        self.model_id = "openai/whisper-large-v3"

    def transcript_audio_to_text(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        processor = AutoProcessor.from_pretrained(self.model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device="cpu",
        )

        current_path = os.path.dirname(os.path.abspath(__file__))

        self.result = pipe(
            f"{current_path}/audio/Gravando2.mp3", return_timestamps=True
        )
        print(self.result["text"])
        print(self.result["chunks"])

    def summarize_audio(self):

        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        ARTICLE = self.result["text"]
        print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

        model_name = "deepset/roberta-base-squad2"

        nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
        QA_input = {
            "question": "Why is model conversion important?",
            "context": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.",
        }
        res = nlp(QA_input)

        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    transcript = Transcript()
    transcript.transcript_audio_to_text()
    # transcript.summarize_audio()