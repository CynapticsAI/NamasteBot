import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import whisper
import edge_tts
import asyncio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub import AudioSegment
from io import BytesIO
import simpleaudio as sa
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import ChatMessage
from langchain.prompts import PromptTemplate

openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)

data_path='your-text-path-here'

chat_history = [] 
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
async def generate_and_play_speech(text):
    """TTS Generation and immediate playback"""
    VOICE = "en-US-ChristopherNeural"
    RATE = "+10%"
    PITCH = "+5Hz"
    
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    
    # Stream directly to memory instead of file
    mp3_data = BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data.write(chunk["data"])
    
    # Reset buffer position
    mp3_data.seek(0)
    
    # Load and play audio directly from memory
    audio = AudioSegment.from_file(mp3_data, format="mp3")
    wave_obj = sa.WaveObject(
        audio.raw_data, 
        num_channels=audio.channels,
        bytes_per_sample=audio.sample_width,
        sample_rate=audio.frame_rate
    )
    play_obj = wave_obj.play()
    play_obj.wait_done()

def store_chat_history(user_message, bot_response):
    chat_history.append(ChatMessage(role="user", content=user_message))
    chat_history.append(ChatMessage(role="assistant", content=bot_response))

def reset_chat_history():
    global chat_history
    chat_history=[]

# Used the above made functions here
def process_query(query):
    """QA Processing"""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    documents=get_txt(data_path)

    texts=split_text(documents)

    vectorstore=vector_store(texts,embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    prompt_template = """ 
    1) Answer the question as detailed as possible from the provided context (at least 100 words) including all available information.
    2) If the exact answer is not found in the documents, expand on relevant topics using your creativity.
    3) If a question has answer in context and in general focus on answering from context.
    4) If the answer is not in the context:
       - Answer without relying on the context.
       - Start the answer with: "Answering in general as context is not provided:"
      
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}  
    )
    
    chat_context = " ".join([f"{message['role']}: {message['content']}" for message in chat_history])
    full_query = chat_context + " user: " + query
    
    response = qa_chain.invoke({"query": full_query})
    
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

    if query=="Bye and clear chat.":
        reset_chat_history()
        
    # Process query
    try:
        response = process_query(query)
        print(f"\n🤖 Response: {response}")
        #Store chat history
        store_chat_history(query, response)
    except Exception as e:
        print(f"\n❌ Error processing query: {str(e)}")
        return
    
    # Generate and play speech
    text = response.replace("**", " ").replace("*", " ").replace("_", " ")
    asyncio.run(generate_and_play_speech(text)) 
   
if __name__ == "__main__":
    main()

