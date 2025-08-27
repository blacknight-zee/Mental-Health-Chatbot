🧠 Mental Health Chatbot with Document-Based QA:  
An intelligent and empathetic mental health chatbot powered by LLaMA 3 (70B) via Groq and LangChain. It uses Retrieval-Augmented Generation (RAG) to answer questions with both emotional intelligence and factual support drawn from PDF documents.

🚀 Features:  
💬 Conversational chatbot for mental health support

🧠 Uses LLaMA 3 (via Groq) for high-quality natural language understanding
📄 Retrieves and processes contextual information from uploaded PDF resources
🧭 Custom prompt and query router to distinguish casual vs. mental health queries
💡 Retrieval-Augmented Generation (RAG) with ChromaDB and HuggingFace embeddings
🔁 Session-based interaction loop

🛠 Technologies Used:  
1. LangChain
2. Groq API (llama3-70b-8192)
3. ChromaDB (Vector Store)
4. Sentence-Transformers (all-MiniLM-L6-v2)
5. PyPDFLoader
6. Prompt Engineering

📄 How It Works:  
1. Vector Store Creation  
Loads PDF files from /content/data  
Splits text into chunks using RecursiveCharacterTextSplitter  
Embeds text using HuggingFace's all-MiniLM-L6-v2  
Stores vectors in a persistent ChromaDB  

2. Chatbot Setup
Initializes the LLaMA 3 model via Groq with a fixed temperature (0)  
Loads the vector store  
Sets up a RetrievalQA chain with a custom prompt for compassionate responses  

3. Chat Loop  
Classifies input as casual or support-seeking  
For casual chats (e.g., "hello", "thanks"), responds directly using the LLM  
For support queries, uses the RetrievalQA chain to answer based on embedded PDF context  

✅ How to Use(Colab):  
Place your mental health-related PDF(s) in the data/ directory.  
Replace the placeholder groq_api_key in the script with your actual key.  
Run the chatbot.  
Start chatting! Type "exit" anytime to end the session.  

Example Interaction:  
Human: hi  
Chatbot: Hi! It's nice to meet you. Is there something I can help you with or would you like to chat?  

Human: I'm feeling stressed about work.  
Chatbot: I'm so sorry to hear that you're reaching out about work-related stress. It takes a lot of courage to acknowledge when things are getting tough. Can you tell me more about what's been going on at work that's causing you stress?  
  
Human: My boss is toxic.  
Chatbot: Having a toxic boss can be incredibly stressful and affect your well-being. You're not alone. Can you share more about what's happening?  


