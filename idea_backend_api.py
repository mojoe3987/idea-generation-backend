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
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
import threading
from collections import Counter
import re

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Qualtrics

# Initialize API client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configuration
CONFIG = {
    'model_name': 'gpt-4o-mini',
    'temperature': 0.8,
    'max_tokens': 800,
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

# Lock for thread-safe updates
state_lock = threading.Lock()

def generate_idea_baseline(topic: str) -> str:
    """Condition 1: No memory"""
    prompt = f"Generate a creative solution for {topic}. Be specific and concrete."
    
    response = openai_client.chat.completions.create(
        model=CONFIG['model_name'],
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    return response.choices[0].message.content.strip()

def generate_idea_with_memory(topic: str, memory_context: str) -> str:
    """Condition 2: Uses shared memory summary"""
    prompt = f"""Generate a creative solution for {topic}.

Previous explorations by other participants:
{memory_context}

Provide a novel approach that explores different territory from what's been covered. Be specific and concrete."""

    response = openai_client.chat.completions.create(
        model=CONFIG['model_name'],
        messages=[
            {"role": "system", "content": "You are a creative idea generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    return response.choices[0].message.content.strip()

def generate_idea_with_exclusion(topic: str, memory_context: str, all_ideas: List[str], 
                                user_request: Optional[str] = None) -> str:
    """Condition 3: Uses shared memory + exclusion, but user can override"""
    excluded_concepts = extract_key_concepts_simple(all_ideas)
    exclusion_text = ", ".join(excluded_concepts[:8])
    
    if user_request:
        # User provided specific direction - honor it even if it overlaps with exclusions
        prompt = f"""Generate a creative solution for {topic}.

Previous explorations by other participants:
{memory_context}

User's specific request:
{user_request}

Note: Other participants have explored: {exclusion_text}
However, honor the user's request above, even if it relates to these concepts. Be specific and concrete."""
    else:
        # Standard exclusion mode
        prompt = f"""Generate a creative solution for {topic}.

Previous explorations by other participants:
{memory_context}

IMPORTANT - You should AVOID these overused concepts (used by other participants):
{exclusion_text}

EXCEPTION: If the user specifically requests an idea related to these concepts, you should honor their request.

Create something using COMPLETELY DIFFERENT concepts, materials, or approaches not listed above. Be specific and concrete."""

    response = openai_client.chat.completions.create(
        model=CONFIG['model_name'],
        messages=[
            {"role": "system", "content": "You are a creative idea generator. You avoid repeating what others have done, but always honor specific user requests."},
            {"role": "user", "content": prompt}
        ],
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    return response.choices[0].message.content.strip()

def extract_key_concepts_simple(ideas: List[str]) -> List[str]:
    """Extract overused concepts from all ideas"""
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                   'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 
                   'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                   'does', 'did', 'will', 'would', 'could', 'should', 'table', 'dining'}
    
    all_words = []
    for idea_obj in ideas:
        idea_text = idea_obj['text'] if isinstance(idea_obj, dict) else idea_obj
        words = re.findall(r'\b[a-z]{4,}\b', idea_text.lower())
        all_words.extend([w for w in words if w not in common_words])
    
    most_common = Counter(all_words).most_common(15)
    return [word for word, count in most_common if count > 2]  # Must appear 3+ times

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
        'ideas_generated': 0,
        'created_at': datetime.now().isoformat()
    }
    
    # Get current state of their condition
    with state_lock:
        state = condition_states[condition]
        total_ideas = len(state['ideas'])
    
    return jsonify({
        'session_id': session_id,
        'condition': condition,
        'topic': topic,
        'total_ideas_in_condition': total_ideas,
        'status': 'success'
    })

@app.route('/api/generate_idea', methods=['POST'])
def generate_idea():
    """Generate idea - adds to SHARED condition state"""
    data = request.json
    session_id = data.get('session_id')
    user_input = data.get('user_input')  # Optional: user's specific request
    
    if session_id not in participant_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = participant_sessions[session_id]
    condition = session['condition']
    topic = session['topic']
    
    try:
        with state_lock:
            state = condition_states[condition]
            current_summary = state['summary']
            all_ideas = state['ideas']
        
        # Generate idea based on condition
        if condition == 'baseline':
            idea = generate_idea_baseline(topic)
        elif condition == 'memory':
            if current_summary:
                idea = generate_idea_with_memory(topic, current_summary)
            else:
                idea = generate_idea_baseline(topic)
        elif condition == 'exclusion':
            if current_summary:
                idea = generate_idea_with_exclusion(
                    topic, 
                    current_summary, 
                    all_ideas,
                    user_request=user_input
                )
            else:
                idea = generate_idea_baseline(topic)
        
        # Add to SHARED state
        with state_lock:
            state['ideas'].append({
                'text': idea,
                'participant_id': session['participant_id'],
                'timestamp': datetime.now().isoformat(),
                'user_request': user_input if user_input else None
            })
            
            total_ideas = len(state['ideas'])
            session['ideas_generated'] += 1
            
            # Update summary every batch_size ideas (across ALL participants)
            if condition in ['memory', 'exclusion'] and total_ideas % CONFIG['batch_size'] == 0:
                batch_start = max(0, total_ideas - CONFIG['batch_size'])
                batch_ideas = [item['text'] for item in state['ideas'][batch_start:]]
                state['summary'] = generate_summary(batch_ideas, state['summary'])
                state['last_summary_update'] = total_ideas
        
        return jsonify({
            'idea': idea,
            'participant_idea_number': session['ideas_generated'],
            'total_ideas_in_condition': total_ideas,
            'session_id': session_id,
            'status': 'success'
        })
        
    except Exception as e:
        app.logger.error(f"Error generating idea: {e}")
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
    
    # Save participant data
    filename = f"participant_{session['participant_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('data/participants', exist_ok=True)
    
    with open(f"data/participants/{filename}", 'w') as f:
        json.dump(session, f, indent=2)
    
    return jsonify({
        'status': 'success',
        'ideas_generated': session['ideas_generated']
    })

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

