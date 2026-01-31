import json

def load_data(file_path):
    """Load the JSON data from the file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def filter_sessions_by_intent(data, intent):
    """Filter sessions by a specific intent."""
    return [session for session in data if session['intent'] == intent]

def filter_sessions_by_service(data, service):
    """Filter sessions by a specific service."""
    return [session for session in data if session['service'] == service]

def extract_confirmed_sessions(data):
    """Extract sessions that have confirmed intents."""
    return [session for session in data if session.get('confirmation_state')]

def get_session(data, session_id):
    """Get a specific session."""
    for session in data['sessions']:
        if session['session_id'] == session_id:
            return session
    return None

def summarize_session_details(session):
    """Summarize the details of a session."""
    summary = {
        'session_id': session['session_id'],
        'intent': session['intent'],
        'service': session['service'],
        'turns': len(session['turns']),
        'confirmed': bool(session.get('confirmation_state'))
    }
    return summary

def summarize_all_sessions(data):
    """Summarize details of all sessions."""
    return [summarize_session_details(session) for session in data]

def get_session_dialogue(session):
    """Get dialogue history of a session."""
    dialogue = []
    for turn in session['turns']:
        dialogue.append({turn['speaker']: turn['utterance']})
    return dialogue
