UPDATED FILE: emotion recognition and sentiment analysis.

🧠Advanced Mental Health Chatbot with Emotional Intelligence

An empathetic mental health chatbot enhanced with emotion & sentiment detection, intent classification, and personalized recommendations. Powered by LLaMA 3 (Groq) and LangChain, it provides supportive conversations and factual guidance.


🚀 Features:  

💬 Conversational chatbot for mental health support

🧠 Uses LLaMA 3 (via Groq) for advanced natural language understanding

😊 Emotion & sentiment analysis for empathetic responses

🧭 Contextual memory for multi-turn conversations

💡 Personalized recommendations for self-care and resources

📄 Retrieval-Augmented Generation (RAG) with ChromaDB + HuggingFace embeddings



🛠 Technologies Used:  
  1.LangChain
  
  2.Groq API (LLaMA 3 - 70B)
  
  3.ChromaDB (Vector Store)
  
  4.Sentence-Transformers (all-MiniLM-L6-v2)
  
  5.HuggingFace Transformers (sentiment & emotion models)
  
  6.PyPDFLoader for document ingestion
  
  7.Prompt Engineering & Conversation Memory
  

📄 How It Works:

1. Document Processing:
PDFs from /content/data are loaded

Text is split into chunks with RecursiveCharacterTextSplitter

Embedded using all-MiniLM-L6-v2

Stored in persistent ChromaDB


2. Emotion & Sentiment Detection

HuggingFace models classify user messages into emotions (joy, anger, sadness, fear, etc.)

Sentiment model classifies into positive, negative, neutral


3. Chatbot Response Flow

Casual → Direct empathetic reply from LLM

Support query → Uses RAG chain for evidence-based guidance

Emotion & sentiment → Tailors response tone


4. Personalized Recommendations
   
Suggests relaxation techniques, exercises, or helpline resources based on detected intent + emotion + sentiment


✅ How to Use (Colab):
1. Add your mental health PDFs in the data/ directory
2. Replace groq_api_key in script with your key
3. Run the notebook
4. Start chatting! (Type exit to quit)

💬 Example Interaction:

Human: hi
Chatbot: Hi! It's really nice to connect with you. How are you feeling today?

Human: I'm feeling very anxious lately.
Chatbot: I hear you. Anxiety can feel overwhelming, but you're not alone. Would you like me to share some calming techniques or explore what’s been triggering this anxiety?

Human: Yes, please.
Chatbot: Sure. One method many people find helpful is deep breathing—inhale slowly for 4 seconds, hold for 4, exhale for 6. Would you like me to also suggest some guided meditation resources?
