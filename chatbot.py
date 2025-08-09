#!/usr/bin/env python3
"""
Chatbot Open-Source Locale con Ollama
Addestra un chatbot su un manuale PDF e lo espone tramite API REST
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
import PyPDF2
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from datetime import datetime

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Classe per processare e estrarre testo dai PDF"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Estrae tutto il testo da un file PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                logger.info(f"Estratto testo da {len(pdf_reader.pages)} pagine")
                return text
        except Exception as e:
            logger.error(f"Errore nell'estrazione del PDF: {e}")
            raise
    
    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Divide il testo in chunks per il training"""
        # Pulisce il testo
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Divide in frasi
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Creati {len(chunks)} chunks di testo")
        return chunks

class VectorStore:
    """Classe per gestire l'embedding e la ricerca semantica"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
    
    def create_embeddings(self, chunks: List[str]):
        """Crea embeddings per tutti i chunks"""
        logger.info("Creazione embeddings...")
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks)
        logger.info(f"Creati embeddings per {len(chunks)} chunks")
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Cerca i chunks più simili alla query"""
        if self.embeddings is None:
            return []
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Ottieni i top_k risultati
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def save(self, filepath: str):
        """Salva il vector store su disco"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'model_name': self.model._modules['0'].auto_model.config.name_or_path
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Vector store salvato in {filepath}")
    
    def load(self, filepath: str):
        """Carica il vector store da disco"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = np.array(data['embeddings']) if data['embeddings'] else None
        logger.info(f"Vector store caricato da {filepath}")

class OllamaClient:
    """Client per interagire con Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def is_model_available(self, model_name: str) -> bool:
        """Verifica se il modello è disponibile in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                return model_name in available_models
            return False
        except Exception as e:
            logger.error(f"Errore nel verificare i modelli disponibili: {e}")
            return False
    
    def pull_model(self, model_name: str):
        """Scarica un modello se non è disponibile"""
        if not self.is_model_available(model_name):
            logger.info(f"Scaricamento modello {model_name}...")
            response = requests.post(f"{self.base_url}/api/pull", 
                                   json={"name": model_name})
            if response.status_code != 200:
                raise Exception(f"Errore nel scaricare il modello: {response.text}")
    
    def generate_response(self, prompt: str, model: str = "mistral") -> str:
        """Genera una risposta usando Ollama"""
        try:
            logger.info(f"Chiamata Ollama con modello: {model}")
            logger.info(f"Prompt length: {len(prompt)} caratteri")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 512,
                        "top_p": 0.9
                    }
                },
                timeout=60*5  # Timeout di 60 secondi
            )
            
            logger.info(f"Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Risposta generata: {len(result.get('response', ''))} caratteri")
                return result.get("response", "Nessuna risposta generata.")
            else:
                logger.error(f"Errore Ollama ({response.status_code}): {response.text}")
                return f"Errore Ollama: {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error("Timeout nella chiamata a Ollama")
            return "Timeout nella generazione della risposta. Riprova."
        except requests.exceptions.ConnectionError:
            logger.error("Impossibile connettersi a Ollama")
            return "Ollama non disponibile. Verifica che sia in esecuzione."
        except Exception as e:
            logger.error(f"Errore nella chiamata a Ollama: {e}")
            return f"Errore: {str(e)}"

class TechnicalChatbot:
    """Chatbot principale che combina ricerca semantica e generazione"""
    
    # def __init__(self, ollama_model: str = "llama2"):
    # italiano
    def __init__(self, ollama_model: str = "mistral"):
        self.vector_store = VectorStore()
        self.ollama_client = OllamaClient()
        self.ollama_model = ollama_model
        self.is_trained = False
    
    def train_from_pdf(self, pdf_path: str, save_path: str = "chatbot_data.pkl"):
        """Addestra il chatbot da un file PDF"""
        logger.info(f"Inizio training da {pdf_path}")
        
        # Verifica che Ollama sia disponibile
        if not self.ollama_client.is_model_available(self.ollama_model):
            logger.info(f"Scaricamento modello {self.ollama_model}...")
            self.ollama_client.pull_model(self.ollama_model)
        
        # Estrai testo dal PDF
        pdf_processor = PDFProcessor()
        text = pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Crea chunks
        chunks = pdf_processor.split_into_chunks(text)
        
        # Crea embeddings
        self.vector_store.create_embeddings(chunks)
        
        # Salva i dati
        self.vector_store.save(save_path)
        self.is_trained = True
        
        logger.info("Training completato!")
    
    def load_training_data(self, data_path: str = "chatbot_data.pkl"):
        """Carica i dati di training salvati"""
        if os.path.exists(data_path):
            self.vector_store.load(data_path)
            self.is_trained = True
            logger.info("Dati di training caricati")
        else:
            logger.warning("File di training non trovato")
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """Genera una risposta alla query dell'utente"""
        logger.info(f"Query ricevuta: {query}")
        
        if not self.is_trained:
            return {
                "response": "Il chatbot non è stato ancora addestrato. Carica un manuale PDF.",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Cerca informazioni rilevanti
        logger.info("Ricerca informazioni rilevanti...")
        relevant_chunks = self.vector_store.search_similar(query, top_k=3)
        logger.info(f"Trovati {len(relevant_chunks)} chunks rilevanti")
        
        if relevant_chunks:
            logger.info(f"Migliore similarity score: {relevant_chunks[0]['similarity']:.3f}")
        
        if not relevant_chunks or relevant_chunks[0]['similarity'] < 0.2:  # Soglia più bassa
            logger.warning("Nessun contenuto rilevante trovato")
            return {
                "response": "Non ho trovato informazioni rilevanti nel manuale per rispondere alla tua domanda. Prova a riformulare la domanda o usa termini più specifici.",
                "sources": [],
                "similarity_scores": [chunk['similarity'] for chunk in relevant_chunks[:3]],
                "timestamp": datetime.now().isoformat()
            }
        
        # Costruisci il context per Ollama
        context = "\n\n".join([f"SEZIONE {i+1}:\n{chunk['text']}" 
                              for i, chunk in enumerate(relevant_chunks)])
        
        # Prompt più semplice e diretto
        prompt = f"""Sei un assistente tecnico. Rispondi alla domanda basandoti SOLO sulle informazioni fornite.

INFORMAZIONI DAL MANUALE:
{context}

DOMANDA: {query}

RISPOSTA (in italiano, massimo 200 parole):"""

        logger.info(f"Invio prompt a Ollama (lunghezza: {len(prompt)})")
        
        # Genera risposta con Ollama
        response = self.ollama_client.generate_response(prompt, self.ollama_model)
        
        logger.info(f"Risposta ricevuta: {len(response)} caratteri")
        
        return {
            "response": response.strip(),
            "sources": [{"text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'], 
                        "similarity": round(chunk['similarity'], 3)} 
                       for chunk in relevant_chunks],
            "similarity_scores": [chunk['similarity'] for chunk in relevant_chunks],
            "timestamp": datetime.now().isoformat()
        }

# Inizializzazione del chatbot globale
chatbot = TechnicalChatbot()

# Flask App
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint di health check"""
    return jsonify({
        "status": "healthy",
        "trained": chatbot.is_trained,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train_chatbot():
    """Endpoint per addestrare il chatbot con un PDF"""
    try:
        data = request.get_json()
        if not data or 'pdf_path' not in data:
            return jsonify({"error": "pdf_path richiesto"}), 400
        
        pdf_path = data['pdf_path']
        if not os.path.exists(pdf_path):
            return jsonify({"error": "File PDF non trovato"}), 404
        
        chatbot.train_from_pdf(pdf_path)
        
        return jsonify({
            "message": "Training completato con successo",
            "trained": True,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel training: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principale per le domande al chatbot"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "query richiesta"}), 400
        
        query = data['query']
        response = chatbot.get_response(query)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Errore nella chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint per ottenere lo status del chatbot"""
    return jsonify({
        "trained": chatbot.is_trained,
        "model": chatbot.ollama_model,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Carica i dati di training se esistono
    chatbot.load_training_data()
    
    logger.info("Avvio server chatbot su http://localhost:9091")
    app.run(host='0.0.0.0', port=9091, debug=False)
