import os
import torch
import whisper
import faiss
import pyttsx3
import speech_recognition as sr
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# === CONFIG ===
TEXT_FILE = "ai.txt"  # 👈 Set your text file here
EMBED_DIM = 384
FAISS_INDEX_FILE = "embeddings/index.faiss"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bigscience/bloom-560m"

# === INIT MODELS ===
print("🔄 Loading Whisper STT...")
stt_model = whisper.load_model("base")

print("🔄 Loading Sentence Transformer...")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("🔄 Loading bigscience/bloom-560m...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("🔄 Initializing pyttsx3 TTS...")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

# === AUDIO RECORDING ===
def record_voice(filename="input.wav"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎙️ Speak now...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    print("✅ Audio saved as", filename)

# === STT ===
def speech_to_text(audio_path="input.wav"):
    print("🧠 Transcribing voice...")
    result = stt_model.transcribe(audio_path)
    return result["text"]

# === DOCUMENT EMBEDDING ===
def index_documents():
    if not os.path.exists(TEXT_FILE):
        print(f"❌ File not found: {TEXT_FILE}")
        return [], None

    with open(TEXT_FILE, encoding='utf-8') as f:
        content = f.read()

    texts = [content]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)

    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)

    print(f"📚 Indexed 1 document: {TEXT_FILE}")
    return texts, embeddings

# === RETRIEVE RELEVANT DOCS ===
def retrieve_documents(query, texts, top_k=1):
    index = faiss.read_index(FAISS_INDEX_FILE)
    query_emb = embed_model.encode([query])
    distances, indices = index.search(query_emb, top_k)
    return [texts[i] for i in indices[0] if i < len(texts)]

# === GENERATE RESPONSE ===
def generate_response(context, query):
    prompt = f"<s>[INST] Given the following context, answer the user's query.\n\nContext:\n{context}\n\nQuery: {query} [/INST]"
    result = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = result[0]["generated_text"].split("[/INST]")[-1].strip()
    return response

# === TTS ===
def speak_text(text):
    print("🗣️ Speaking...")
    tts_engine.say(text)
    tts_engine.runAndWait()

# === MAIN APP ===
def main():
    texts, _ = index_documents()
    if not texts:
        return

    record_voice()
    query = speech_to_text("input.wav")
    print(f"\n🧏 You said: {query}\n")

    context = "\n\n".join(retrieve_documents(query, texts))
    response = generate_response(context, query)

    print(f"🤖 Assistant:\n{response}\n")
    speak_text(response)

if __name__ == "__main__":
    main()