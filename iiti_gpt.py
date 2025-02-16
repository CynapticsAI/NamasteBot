!pip install -q --upgrade google-generativeai langchain-google-genai  langchain-community
!pip install git+https://github.com/openai/whisper.git
!pip install faiss-cpu
!pip install edge-tts
!pip install pydub -q
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from IPython.display import HTML, Javascript, Audio
from google.colab import output
from base64 import b64decode
from pydub import AudioSegment
import edge_tts
import whisper
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",credentials=credentials)
loader = TextLoader("/content/drive/MyDrive/IITI GPT/last data.txt")
documents = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",credentials=credentials)
vectorstore = FAISS.from_documents(documents, embeddings)
# Define JavaScript/HTML interface
js_code = """
<div id="recorder">
  <button onclick="startRecording()" id="startBtn">🎤 Start Recording</button>
  <button onclick="stopRecording()" id="stopBtn" disabled>⏹️ Stop & Save</button>
</div>

<script>
let mediaRecorder;
let audioChunks = [];

async function startRecording() {
  document.getElementById("stopBtn").disabled = false;
  document.getElementById("startBtn").disabled = true;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.start();
  } catch (error) {
    alert('Error accessing microphone: ' + error.message);
  }
}

async function stopRecording() {
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;

  mediaRecorder.stop();
  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const reader = new FileReader();

    reader.readAsDataURL(audioBlob);
    reader.onloadend = () => {
      const base64data = reader.result;
      google.colab.kernel.invokeFunction('notebook.saveAudio', [base64data], {});
    };

    audioChunks = [];
  };
}
</script>
"""
def save_audio(base64_data):
    try:
        # Decode base64 audio data
        audio_bytes = b64decode(base64_data.split(',')[1])

        # Save original webm file
        with open('my_recording.webm', 'wb') as f:
            f.write(audio_bytes)

        # Convert to WAV format
        audio = AudioSegment.from_file('my_recording.webm', format='webm')
        audio.export('my_recording.wav', format='wav')

        # Show audio player
        display(Audio('my_recording.wav'))
        print("✅ Audio saved as my_recording.wav")

    except Exception as e:
        print("Error processing audio:", e)

# Register callback
output.register_callback('notebook.saveAudio', save_audio)

# Display the recorder interface
display(HTML(js_code))


query=whisper.load_model("medium").transcribe("my_recording.wav",language="en")['text']
print(query)


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
response = qa_chain.invoke({"query": query})
print("\nGenerated Response:\n", response["result"])
# Your text
text = response["result"]
text = text.replace("**", " ")
text = text.replace("*", " ")
text = text.replace("_", " ")
# Customize voice and speech parameters
VOICE = "en-US-ChristopherNeural"  # Deep male voice
RATE = "+10%"  # Speed adjustment (-20% to +20% for natural pacing)
PITCH = "+5Hz"  # Slightly lower pitch for warmth

# Generate speech with adjustments
async def generate_speech():
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    await communicate.save("human_like_audio.mp3")

# Run in Colab/Jupyter (no asyncio loop needed)
await generate_speech()

# Auto-play enhanced audio
Audio("human_like_audio.mp3", autoplay=True)
