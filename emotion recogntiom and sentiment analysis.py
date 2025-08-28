

from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def initialize_analyzers():
    print("Loading emotion and sentiment analysis models...")

    emotion_analyzer = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-roberta-large",
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )


    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )

    print("Models loaded successfully!")
    return emotion_analyzer, sentiment_analyzer


def analyze_text(text, emotion_analyzer, sentiment_analyzer):

    emotions = emotion_analyzer(text)[0]

    emotions.sort(key=lambda x: x['score'], reverse=True)
    dominant_emotion = emotions[0]['label']


    sentiment = sentiment_analyzer(text)[0]

    return {
        'dominant_emotion': dominant_emotion,
        'emotion_details': emotions,
        'sentiment': sentiment['label'],
        'sentiment_score': sentiment['score']
    }


def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192"
    )
    return llm


def create_vector_db():
    loader = DirectoryLoader("/content/data", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    print("ChromaDB created and data saved")
    return vector_db


def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health support companion. Respond with warmth, empathy, and understanding.
    Consider the emotional context provided, but don't explicitly mention it. Instead, weave that understanding into a natural,
    supportive response that may include gentle suggestions for coping strategies, mindfulness exercises, or words of encouragement
    when appropriate. Keep responses conversational yet professional.

    Context: {context}
    User's message: {question}

    Respond in a way that:
    - Acknowledges their feelings implicitly
    - Offers genuine understanding and validation
    - Provides gentle guidance or suggestions when relevant
    - Maintains a warm, supportive tone
    - Ends with hope or encouragement
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={'prompt': PROMPT}
    )
    return qa_chain


def is_general_chat(query):
    general_keywords = ["hi", "hello", "hey", "how are you", "good morning", "good evening", "thank you", "thanks", "yo", "greetings"]
    return any(keyword in query.lower() for keyword in general_keywords)


def main():
    print("Initializing Chatbot.........")
    llm = initialize_llm()


    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Using {device} for emotion and sentiment analysis")

    emotion_analyzer, sentiment_analyzer = initialize_analyzers()

    db_path = "/content/chroma_db"

    if not os.path.exists(db_path):
        vector_db = create_vector_db()
    else:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    qa_chain = setup_qa_chain(vector_db, llm)

    print("\nI'm here to listen and support you. Feel free to share whatever is on your mind.")

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("\nThank you for sharing with me today. Remember to be gentle with yourself, and know that seeking support is a sign of strength. Take care, and I'm here whenever you need to talk.")
            break


        analysis = analyze_text(query, emotion_analyzer, sentiment_analyzer)

        if is_general_chat(query):
            prompt = f"""As a compassionate mental health companion, respond to: '{query}'
            Context: The person is feeling {analysis['dominant_emotion']} with a {analysis['sentiment']} sentiment.
            Create a warm, natural response that validates their experience without explicitly naming emotions.
            If appropriate, gently suggest a simple mindfulness technique or coping strategy."""

            response = llm.invoke(prompt)
            print(f"\n{response.content}")
        else:
            enhanced_query = f"""Understanding the emotional context - {analysis['dominant_emotion']} with
            {analysis['sentiment']} sentiment - respond to: {query}
            Provide a supportive response that validates their experience and offers gentle guidance when appropriate."""

            response = qa_chain.invoke({"query": enhanced_query})
            print(f"\n{response['result']}")


if __name__ == "__main__":
    main()
