"""
Backend API for Qualtrics-Integrated Idea Diversity Experiment
============================================================
Manages shared memory across participants in three conditions:
1. Baseline: No memory
2. Memory: Shared summaries
3. Exclusion: Shared summaries + avoid overused concepts

Author: Research Team
Date: October 2025
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
from typing import List, Dict, Optional, Tuple
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
import threading
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine, pdist

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Qualtrics

# Initialize API client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configuration
CONFIG = {
    'model_name': 'gpt-4o-mini',
    'temperature': 0.8,
    'max_tokens': 800,  # Enough for complete 2-3 sentence ideas, no cutoffs
    'batch_size': 10  # Update summary every 10 ideas across all participants
}

# GLOBAL SHARED STATE - separate for each condition
# This accumulates across ALL participants
condition_states = {
    'baseline': {
        'ideas': [],
        'summary': None,
        'last_summary_update': 0
    },
    'memory': {
        'ideas': [],
        'summary': None,
        'last_summary_update': 0
    },
    'exclusion': {
        'ideas': [],
        'summary': None,
        'last_summary_update': 0
    }
}

# Individual participant sessions (just for tracking)
participant_sessions = {}

# Embedding cache - stores computed embeddings by text hash
# This provides identical results but much faster for repeated/similar ideas
embedding_cache = {}
MAX_CACHE_SIZE = 1000  # Limit cache to prevent memory issues

# Lock for thread-safe updates
state_lock = threading.Lock()

def build_system_prompt(condition: str, topic: str, memory_summary: Optional[str], all_ideas: List[str]) -> str:
    """Build condition-specific system prompt for natural conversation"""
    
    base_prompt = f"""You are a helpful AI assistant having a natural conversation to help users brainstorm creative ideas for {topic}.

- Respond naturally and conversationally
- When the user asks for an idea or makes a request, provide ONE specific idea in 2-3 sentences
- Describe ideas directly
- Do not use imperative verbs like "Create", "Design", or "Introduce"
- If the user greets you or chats, respond warmly and guide them towards idea generation"""
    
    if condition == 'baseline':
        return base_prompt
    
    elif condition == 'memory':
        # Add memory context invisibly
        if memory_summary and len(all_ideas) >= 2:
            memory_context = f"\n\nCONTEXT (not visible to user): Other participants have explored:\n{memory_summary}"
            return base_prompt + memory_context
        return base_prompt
    
    elif condition == 'exclusion':
        # Add memory + exclusion context invisibly
        if memory_summary and len(all_ideas) >= 2:
            exclusion_context = f"\n\nCONTEXT (not visible to user): Other participants have explored:\n{memory_summary}"
            exclusion_context += "\n\nIMPORTANT: When generating ideas, avoid ideas similar in MEANING to the patterns described above."
            exclusion_context += "\nHOWEVER: If the user specifically requests something related to these patterns, honor their request."
            
            return base_prompt + exclusion_context
        return base_prompt
    
    return base_prompt

def generate_idea_baseline(topic: str, user_request: str) -> str:
    """Condition 1: User request only, no memory"""
    prompt = f"""Generate a creative solution for {topic}. Keep it concise (2-3 sentences). Be specific and concrete.

User's request:
{user_request}"""
    
    response = openai_client.chat.completions.create(
        model=CONFIG['model_name'],
        messages=[
            {"role": "system", "content": "You are a creative idea generator. Describe ideas directly. Do not use imperative verbs like 'Create', 'Design', or 'Introduce'. Keep responses to 2-3 sentences."},
            {"role": "user", "content": prompt}
        ],
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    return response.choices[0].message.content.strip()

def generate_idea_with_memory(topic: str, memory_context: str, all_ideas: List[str], user_request: str) -> str:
    """Condition 2: Shows summaries only - NO explicit avoidance instruction"""
    # Simplified approach: Only use summaries, no semantic themes or clustering
    
    prompt = f"""Generate a creative solution for {topic}. Keep it concise (2-3 sentences). Be specific and concrete.

Previous explorations by other participants:
{memory_context}

User's request:
{user_request}"""

    response = openai_client.chat.completions.create(
        model=CONFIG['model_name'],
        messages=[
            {"role": "system", "content": "You are a creative idea generator. Describe ideas directly. Do not use imperative verbs like 'Create', 'Design', or 'Introduce'. Keep responses to 2-3 sentences."},
            {"role": "user", "content": prompt}
        ],
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    return response.choices[0].message.content.strip()

def generate_idea_with_exclusion(topic: str, memory_context: str, all_ideas: List[str], 
                                user_request: Optional[str] = None) -> str:
    """Condition 3: Uses summaries + simple avoidance instruction"""
    # Simplified approach: Only use summaries + simple avoidance, no semantic themes or keyword extraction
    
    prompt = f"""Generate a creative solution for {topic}. Keep it concise (2-3 sentences).

Previous explorations by other participants:
{memory_context}

User's specific request:
{user_request}

IMPORTANT: Avoid repeating the approaches and ideas mentioned in the previous explorations above. Be specific and concrete in your idea."""

    response = openai_client.chat.completions.create(
        model=CONFIG['model_name'],
        messages=[
            {"role": "system", "content": "You are a creative idea generator. Describe ideas directly. Do not use imperative verbs like 'Create', 'Design', or 'Introduce'. Keep responses to 2-3 sentences. You avoid repeating what others have done, but always honor specific user requests."},
            {"role": "user", "content": prompt}
        ],
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    return response.choices[0].message.content.strip()

# UNUSED IN SIMPLIFIED APPROACH - Commented out to reduce complexity
# def get_embeddings(texts: List[str]) -> np.ndarray:
#     """Get semantic embeddings for texts using OpenAI with caching"""
#     if not texts:
#         return np.array([])
#     
#     embeddings = []
#     texts_to_fetch = []
#     indices_to_fetch = []
#     
#     # Check cache first
#     for i, text in enumerate(texts):
#         text_hash = hashlib.md5(text.encode()).hexdigest()
#         if text_hash in embedding_cache:
#             embeddings.append(embedding_cache[text_hash])
#         else:
#             embeddings.append(None)
#             texts_to_fetch.append(text)
#             indices_to_fetch.append(i)
#     
#     # Fetch uncached embeddings in batches
#     if texts_to_fetch:
#         try:
#             # Batch up to 100 texts at once (OpenAI limit is 2048)
#             batch_size = 100
#             for batch_start in range(0, len(texts_to_fetch), batch_size):
#                 batch_end = min(batch_start + batch_size, len(texts_to_fetch))
#                 batch = texts_to_fetch[batch_start:batch_end]
#                 
#                 response = openai_client.embeddings.create(
#                     model="text-embedding-3-small",
#                     input=batch
#                 )
#                 
#                 # Cache and store the results
#                 for j, item in enumerate(response.data):
#                     global_idx = batch_start + j
#                     text = texts_to_fetch[global_idx]
#                     text_hash = hashlib.md5(text.encode()).hexdigest()
#                     embedding_cache[text_hash] = item.embedding
#                     embeddings[indices_to_fetch[global_idx]] = item.embedding
#                 
#                 # Limit cache size to prevent memory issues
#                 if len(embedding_cache) > MAX_CACHE_SIZE:
#                     # Remove oldest entries (simple FIFO)
#                     keys_to_remove = list(embedding_cache.keys())[:len(embedding_cache) - MAX_CACHE_SIZE + 100]
#                     for key in keys_to_remove:
#                         del embedding_cache[key]
#                     app.logger.info(f"Cache size limited to {MAX_CACHE_SIZE}, removed {len(keys_to_remove)} entries")
#                 
#             app.logger.info(f"Cached {len(texts_to_fetch)} new embeddings (cache size: {len(embedding_cache)})")
#         except Exception as e:
#             app.logger.error(f"Error getting embeddings: {e}")
#             return np.array([])
#     else:
#         app.logger.info(f"All {len(texts)} embeddings retrieved from cache")
#     
#     return np.array(embeddings)

# UNUSED IN SIMPLIFIED APPROACH - Commented out to reduce complexity
# def extract_semantic_themes(ideas: List[str], n_clusters: int = 5) -> Tuple[List[str], np.ndarray]:
#     """
#     Extract semantic themes using embeddings + clustering.
#     Returns: (theme descriptions, cluster centers)
#     """
#     if len(ideas) < 2:
#         return [], np.array([])
#     
#     try:
#         # Get semantic embeddings
#         embeddings = get_embeddings(ideas)
#         if len(embeddings) == 0:
#             return [], np.array([])
#         
#         # Cluster ideas semantically
#         n_clusters = min(n_clusters, len(ideas) // 2)
#         if n_clusters < 2:
#             return [], np.array([])
#         
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         cluster_labels = kmeans.fit_predict(embeddings)
#         
#         # Get representative ideas from each cluster (closest to center)
#         cluster_themes = []
#         for i in range(n_clusters):
#             cluster_indices = np.where(cluster_labels == i)[0]
#             if len(cluster_indices) == 0:
#                 continue
#             
#             cluster_embeddings = embeddings[cluster_indices]
#             center = kmeans.cluster_centers_[i]
#             
#             # Find idea closest to cluster center
#             distances = [cosine(emb, center) for emb in cluster_embeddings]
#             closest_idx = cluster_indices[np.argmin(distances)]
#             
#             # Use the representative idea (truncated for brevity)
#             representative = ideas[closest_idx][:100] + "..." if len(ideas[closest_idx]) > 100 else ideas[closest_idx]
#             cluster_themes.append(representative)
#         
#         return cluster_themes, kmeans.cluster_centers_
#         
#     except Exception as e:
#         app.logger.warning(f"Error extracting semantic themes: {e}")
#         return [], np.array([])

# UNUSED IN SIMPLIFIED APPROACH - Commented out to reduce complexity
# def extract_key_concepts(ideas: List[str], max_concepts: int = 15) -> List[str]:
#     """Extract key concepts using TF-IDF for keyword-based exclusion"""
#     if len(ideas) < 2:
#         return []
#     
#     try:
#         # Use TF-IDF to find important terms
#         vectorizer = TfidfVectorizer(
#             max_features=max_concepts * 2,
#             stop_words='english',
#             ngram_range=(1, 2),  # Include bigrams (e.g., "modular design")
#             min_df=1
#         )
#         tfidf_matrix = vectorizer.fit_transform(ideas)
#         feature_names = vectorizer.get_feature_names_out()
#         
#         # Get top concepts based on TF-IDF scores
#         scores = tfidf_matrix.sum(axis=0).A1
#         top_indices = scores.argsort()[-max_concepts:][::-1]
#         
#         concepts = [feature_names[i] for i in top_indices]
#         
#         # Filter out very common words that might slip through
#         filtered_concepts = [c for c in concepts if len(c.split()) > 1 or len(c) > 4]
#         
#         return filtered_concepts[:max_concepts]
#         
#     except Exception as e:
#         app.logger.warning(f"Error extracting concepts: {e}")
#         return []

def generate_summary(ideas: List[str], existing_summary: Optional[str] = None) -> str:
    """Generate/update summary of ALL ideas across participants"""
    if existing_summary:
        prompt = f"""Update this summary with new ideas from participants (keep concise, max 12 sentences):

Existing summary:
{existing_summary}

New ideas:
{chr(10).join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])}

Updated summary:"""
    else:
        prompt = f"""Summarize these ideas in 3-4 sentences, focusing on main themes and approaches:

{chr(10).join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])}

Summary:"""
    
    response = openai_client.chat.completions.create(
        model=CONFIG['model_name'],
        messages=[
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Initialize participant session"""
    data = request.json
    participant_id = data.get('participant_id')
    condition = data.get('condition')  # 'baseline', 'memory', 'exclusion'
    topic = data.get('topic', 'innovative dining table designs')
    
    if condition not in ['baseline', 'memory', 'exclusion']:
        return jsonify({'error': 'Invalid condition'}), 400
    
    session_id = hashlib.md5(f"{participant_id}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    participant_sessions[session_id] = {
        'participant_id': participant_id,
        'condition': condition,
        'topic': topic,
        'created_at': datetime.now().isoformat()
    }
    
    # Get current state of their condition
    with state_lock:
        state = condition_states[condition]
        total_ideas = len(state['ideas'])
    
    # Log session start with condition
    app.logger.info(f"NEW SESSION: Participant {participant_id[:8]}... assigned to {condition.upper()} condition (total ideas in condition: {total_ideas})")
    
    return jsonify({
        'session_id': session_id,
        'condition': condition,
        'topic': topic,
        'total_ideas_in_condition': total_ideas,
        'status': 'success'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Natural conversation with condition-specific context injected"""
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message')
    chat_history = data.get('chat_history', [])  # Full conversation history
    
    app.logger.info(f"CHAT REQUEST: session_id={session_id}, message={user_message[:50] if user_message else 'None'}...")
    
    if session_id not in participant_sessions:
        app.logger.error(f"Invalid session: {session_id}. Active sessions: {list(participant_sessions.keys())}")
        return jsonify({'error': 'Session expired. Please refresh the page to start a new session.'}), 400
    
    session = participant_sessions[session_id]
    condition = session['condition']
    topic = session['topic']
    
    try:
        with state_lock:
            state = condition_states[condition]
            current_summary = state['summary']
            all_ideas = state['ideas']
        
        # Build system prompt based on condition
        system_prompt = build_system_prompt(condition, topic, current_summary, all_ideas)
        
        # Build messages with full chat history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in chat_history:
            if msg['sender'] == 'You':
                messages.append({"role": "user", "content": msg['message']})
            elif msg['sender'] == 'Assistant':
                messages.append({"role": "assistant", "content": msg['message']})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Get response
        response = openai_client.chat.completions.create(
            model=CONFIG['model_name'],
            messages=messages,
            temperature=CONFIG['temperature'],
            max_tokens=CONFIG['max_tokens']
        )
        
        assistant_response = response.choices[0].message.content.strip()
        
        # Log conversation interaction
        app.logger.info(f"CHAT: {condition.upper()} condition - Participant {session['participant_id'][:8]}... sent message, received response")
        
        return jsonify({
            'response': assistant_response,
            'session_id': session_id,
            'status': 'success'
        })
        
    except Exception as e:
        app.logger.error(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_condition_stats', methods=['GET'])
def get_condition_stats():
    """Get statistics for all conditions"""
    with state_lock:
        stats = {
            condition: {
                'total_ideas': len(state['ideas']),
                'has_summary': state['summary'] is not None,
                'last_summary_update': state['last_summary_update']
            }
            for condition, state in condition_states.items()
        }
    
    return jsonify(stats)

@app.route('/api/end_session', methods=['POST'])
def end_session():
    """Save participant session data"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in participant_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = participant_sessions[session_id]
    
    # Log session end
    app.logger.info(f"SESSION ENDED: {session['condition'].upper()} - Participant {session['participant_id'][:8]}...")
    
    # Save participant data
    filename = f"participant_{session['participant_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('data/participants', exist_ok=True)
    
    with open(f"data/participants/{filename}", 'w') as f:
        json.dump(session, f, indent=2)
    
    return jsonify({
        'status': 'success'
    })

@app.route('/api/submit_idea', methods=['POST'])
def submit_idea():
    """Receive an idea from Qualtrics QID18 and add it to the appropriate condition"""
    data = request.json
    session_id = data.get('session_id')
    idea_text = data.get('idea_text')
    
    if not session_id or not idea_text:
        return jsonify({'error': 'Missing session_id or idea_text'}), 400
    
    if session_id not in participant_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = participant_sessions[session_id]
    condition = session['condition']
    
    try:
        with state_lock:
            state = condition_states[condition]
            state['ideas'].append({
                'text': idea_text,
                'participant_id': session['participant_id'],
                'timestamp': datetime.now().isoformat(),
                'source': 'qualtrics_qid18'
            })
            
            total_ideas = len(state['ideas'])
            
            # Update summary every batch_size ideas (across ALL participants)
            summary_updated = False
            if condition in ['memory', 'exclusion'] and total_ideas % CONFIG['batch_size'] == 0:
                try:
                    batch_start = max(0, total_ideas - CONFIG['batch_size'])
                    batch_ideas = [item['text'] for item in state['ideas'][batch_start:]]
                    state['summary'] = generate_summary(batch_ideas, state['summary'])
                    state['last_summary_update'] = total_ideas
                    summary_updated = True
                    app.logger.info(f"Summary updated for {condition} condition")
                except Exception as e:
                    app.logger.error(f"Error updating summary: {e}")
        
        # Log idea submission
        app.logger.info(f"IDEA SUBMITTED: {condition.upper()} condition - Participant {session['participant_id'][:8]}... (total in condition: {total_ideas}){' [SUMMARY UPDATED]' if summary_updated else ''}")
        
        return jsonify({
            'status': 'success',
            'total_ideas_in_condition': total_ideas,
            'summary_updated': summary_updated
        })
        
    except Exception as e:
        app.logger.error(f"Error submitting idea: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_condition_data', methods=['GET'])
def export_condition_data():
    """Export all data for a condition (for analysis)"""
    condition = request.args.get('condition')
    
    if condition not in condition_states:
        return jsonify({'error': 'Invalid condition'}), 400
    
    with state_lock:
        state = condition_states[condition]
        data = {
            'condition': condition,
            'total_ideas': len(state['ideas']),
            'ideas': state['ideas'],
            'summary': state['summary'],
            'exported_at': datetime.now().isoformat()
        }
    
    # Save to file
    filename = f"condition_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('data/exports', exist_ok=True)
    
    with open(f"data/exports/{filename}", 'w') as f:
        json.dump(data, f, indent=2)
    
    return jsonify(data)

@app.route('/api/reset_all_data', methods=['POST'])
def reset_all_data():
    """Reset all condition data (for testing/experiment start)"""
    try:
        with state_lock:
            # Reset all conditions
            for condition in condition_states:
                condition_states[condition] = {
                    'ideas': [],
                    'summary': None,
                    'last_summary_update': 0
                }
            
            # Clear participant sessions
            participant_sessions.clear()
            
            # Clear embedding cache
            embedding_cache.clear()
        
        app.logger.info("All data reset - experiment ready to start")
        
        return jsonify({
            'status': 'success',
            'message': 'All condition data and sessions have been reset',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error resetting data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Periodic auto-save (run in background)
def auto_save_state():
    """Save state every 5 minutes"""
    def save():
        while True:
            time.sleep(300)  # 5 minutes
            with state_lock:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"data/backups/state_backup_{timestamp}.json"
                os.makedirs('data/backups', exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(condition_states, f, indent=2)
                app.logger.info(f"Auto-saved state to {filename}")
    
    thread = threading.Thread(target=save, daemon=True)
    thread.start()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/participants', exist_ok=True)
    os.makedirs('data/exports', exist_ok=True)
    os.makedirs('data/backups', exist_ok=True)
    
    # Start auto-save
    auto_save_state()
    
    # Run the app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

