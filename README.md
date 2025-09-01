
# Mental Health Support Bot

## Description

A compassionate AI-powered mental health support chatbot designed to provide empathetic conversations and emotional support. This bot combines advanced natural language processing with mental health expertise to offer personalized responses, emotion recognition, and sentiment analysis. It can analyze user messages to understand emotional states and provide appropriate, supportive responses while maintaining a safe and non-judgmental environment.

## Features

- 🧠 **Emotion Recognition**: Analyzes text to identify emotional states (joy, sadness, anger, fear, surprise, disgust, neutral)
- 💭 **Sentiment Analysis**: Determines positive/negative sentiment with confidence scores
- 🤖 **AI-Powered Responses**: Uses Groq LLM (LLaMA 3 - 70B) for intelligent, supportive responses
- 📚 **Document Retrieval**: Can access knowledge base from mental health PDFs for informed responses
- 🔄 **Conversation Memory**: Maintains context throughout the conversation
- 🎯 **Personalized Support**: Adapts responses based on detected emotions and sentiment
- 📖 **Educational Resources**: Provides information from mental health documents

## Tech Stack

- **LangChain**: Framework for building LLM applications
- **Groq API**: High-performance LLM inference (LLaMA 3 - 70B model)
- **ChromaDB**: Vector database for document storage and retrieval
- **Sentence-Transformers**: Text embeddings (all-MiniLM-L6-v2 model)
- **HuggingFace Transformers**: Pre-trained models for sentiment and emotion analysis
- **PyPDFLoader**: Document ingestion from PDF files
- **Python**: Core programming language

## How It Works

1. **Input Processing**: User messages are analyzed for emotion and sentiment using HuggingFace models
2. **Context Retrieval**: Relevant information is retrieved from the mental health document database
3. **Response Generation**: The Groq LLM generates supportive responses based on:
   - Detected emotion and sentiment
   - Retrieved context from documents
   - Conversation history
   - Mental health best practices
4. **Output**: Provides both the AI response and analysis results

## File Structure

- `mental_health_bot.ipynb` - Original Jupyter notebook with main LLM bot and RAG implementation
- `emotion recogntiom and sentiment analysis.py` - Enhanced implementation with emotion and sentiment analysis
- `requirements.txt` - Python dependencies
- `data/` - Directory containing mental health PDF documents
- `README.md` - This file


## Sample Conversation

```
Human: Hi, I'm feeling really overwhelmed with work lately.

Bot: Hi! I can sense that you're feeling overwhelmed, and I want you to know that's completely understandable. Work stress can be really challenging to manage. 

Would you like to talk about what specifically is making work feel overwhelming? Sometimes breaking it down can help make it feel more manageable.

Human: Yes, I have so many deadlines and I can't seem to focus on anything.

Bot: I hear you, and that sounds really stressful. When we're juggling multiple deadlines, it's common to feel scattered and unable to focus. 


Let me suggest a few strategies that might help:
1. Try the "Pomodoro Technique" - work for 25 minutes, then take a 5-minute break
2. Make a simple to-do list and tackle one task at a time
3. Consider talking to your supervisor about prioritizing tasks

What do you think about trying one of these approaches?

Human: The Pomodoro Technique sounds interesting. How does it work exactly?

Bot: Great question! The Pomodoro Technique is a time management method that can really help with focus and productivity.


Here's how it works:
1. Choose a task you want to work on
2. Set a timer for 25 minutes (this is one "Pomodoro")
3. Work on that task until the timer rings
4. Take a 5-minute break
5. After 4 Pomodoros, take a longer 15-30 minute break

The key is that during those 25 minutes, you commit to working on just that one task - no checking emails, no social media, just focused work. It's amazing how much you can accomplish in short, focused bursts!

Would you like to try it with a specific task you're working on?
```


