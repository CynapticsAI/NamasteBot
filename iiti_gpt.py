from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
import whisper
import edge_tts
import asyncio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tough-dreamer-449219-p2-6a5c6c7ee6a3.json"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

api_key="enter your open api key"   

data_path='C:/Users/switi/OneDrive/Desktop/IITI GPT/data.txt'
# Loading text from txt file
def get_txt(data_path):
    loader = TextLoader(data_path,encoding='utf-8')
    documents = loader.load()
    return documents
# Splitting that text
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts
# assinging them vectors in vector space
def vector_store(texts,embeddings):
    vectorstore = FAISS.from_documents(texts,embeddings)
    return vectorstore
# Didn't alter anything here
def record_audio(filename='recording.wav', samplerate=16000):
    """Record audio until user stops it"""
    print("\nPress Enter to start recording...")
    input()
    
    audio_data = []
    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='int16',
        callback=lambda indata, frames, time, status: audio_data.append(indata.copy())
    )
    
    try:
        print("Recording... Press Enter to stop")
        stream.start()
        input()  
    finally:
        stream.stop()
        stream.close()
        print("⏹️ Recording stopped")
        
        if audio_data:
            audio_array = np.concatenate(audio_data)
            write(filename, samplerate, audio_array)
            print(f"✅ Saved to {filename}")
            return filename
        return None

async def generate_speech_async(text, filename="output.mp3"):
    """TTS Generation"""
    VOICE = "en-US-ChristopherNeural"
    RATE = "+10%"
    PITCH = "+5Hz"
    
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    await communicate.save(filename)
    return filename
# Used the above made functions here
def process_query(query):
    """QA Processing"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 

    documents=get_txt(data_path)

    texts=split_text(documents)

    vectorstore=vector_store(texts,embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]

def main():
    audio_file = record_audio()
    
    if not audio_file:
        print("❌ No audio recorded")
        return
    
    # Transcribe audio
    model = whisper.load_model("small")  # Consider using smaller model if resources are limited
    result = model.transcribe(audio_file, language="en")
    query = result["text"]
    print(f"\n🗣️ Query: {query}")
    
    # Process query
    try:
        response = process_query(query)
        print(f"\n🤖 Response: {response}")
    except Exception as e:
        print(f"\n❌ Error processing query: {str(e)}")
        return
    
    # Generate and play speech
    text = response.replace("**", " ").replace("*", " ").replace("_", " ")
    try:
        asyncio.run(generate_speech_async(text))
        audio = AudioSegment.from_file("output.mp3")
        play(audio)
    except Exception as e:
        print(f"🔇 Playback error: {str(e)}")

if __name__ == "__main__":
    main()
