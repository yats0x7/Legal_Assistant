import os
import random
import joblib
import ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
MODEL_PATH = "models/"

print("--- ⚖️  INITIALIZING INDIAN LEGAL ASSISTANT ---")

# 1. LOAD SENTIMENT & INTENT MODELS (Steps 1 & 2)
print("1️⃣  Loading Classifiers...")
try:
    sent_model = joblib.load(os.path.join(MODEL_PATH, 'sentiment_xgb.pkl'))
    sent_vec = joblib.load(os.path.join(MODEL_PATH, 'sentiment_vectorizer.pkl'))
    
    intent_model = joblib.load(os.path.join(MODEL_PATH, 'intent_model.pkl'))
    intent_vec = joblib.load(os.path.join(MODEL_PATH, 'intent_vectorizer.pkl'))
except FileNotFoundError:
    print("❌ Error: Models not found. Please run train_sentiment.py and train_intent.py first.")
    exit()

# 2. LOAD VECTOR DB (Step 3)
print("2️⃣  Connecting to Legal Knowledge Base...")
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embed_model)

# 3. LOAD SAFETY FILTER (Step 5)
print("3️⃣  Activating Safety Protocols...")
safety_filter = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

print("✅ SYSTEM READY.\n")

# --- GREETING / CASUAL RESPONSES ---

GREETING_RESPONSES = [
    "Hello! 👋 I'm your AI Legal Assistant specializing in Indian law. How can I help you today?",
    "Hey there! 😊 I'm here to help you with any legal questions about Indian law. What's on your mind?",
    "Hi! 🙏 Welcome! I can help you understand Indian laws, legal procedures, and your rights. Ask me anything!",
    "Namaste! 🙏 I'm your legal AI assistant. Feel free to ask me about IPC, BNS, Constitution, or any Indian legal matter.",
    "Hello! 😊 Great to see you! I'm ready to help with any legal queries — from FIR procedures to property disputes. What do you need?",
]

FAREWELL_RESPONSES = [
    "Goodbye! 👋 Take care and don't hesitate to come back if you have more legal questions!",
    "See you later! 😊 Remember, I'm always here if you need legal guidance.",
    "Bye! 🙏 Stay safe and feel free to reach out anytime for legal help.",
    "Take care! 👋 Wishing you the best with everything.",
]

HOW_ARE_YOU_RESPONSES = [
    "I'm doing great, thank you for asking! 😊 I'm ready to help you with any legal questions. What can I do for you?",
    "I'm functioning perfectly! 💪 More importantly, how can I assist you today? Any legal matter you'd like to discuss?",
    "All good on my end! 😊 Thank you for asking. Now tell me, what legal query do you have?",
    "I'm well, thanks! 🙏 Always happy to help. Do you have a legal question for me?",
]

THANKS_RESPONSES = [
    "You're most welcome! 😊 Happy I could help. Let me know if you have any more questions!",
    "Glad I could assist! 🙏 Don't hesitate to ask if anything else comes up.",
    "My pleasure! 😊 I'm always here if you need more legal guidance.",
    "You're welcome! 🙌 Feel free to come back anytime.",
]

ABOUT_ME_RESPONSES = [
    "I'm an AI Legal Assistant 🤖⚖️ trained to help you understand Indian laws! I can help with:\n- 👮 Criminal Law (IPC, BNS, CrPC)\n- 🏠 Family & Civil Law (Property, Divorce, Custody)\n- 💼 Corporate Law (Companies Act, GST, Contracts)\n\nJust ask me a question and I'll do my best to help!",
]

GENERIC_CASUAL_RESPONSES = [
    "I appreciate the chat! 😊 I'm here to help with legal questions — feel free to ask me anything about Indian law!",
    "Nice talking to you! 😊 Whenever you're ready, I can help with any legal queries you might have.",
    "That's nice! 😊 I'm best at helping with legal matters though — got any questions about Indian law?",
]

# Keywords for sub-classifying casual messages
FAREWELL_KEYWORDS = {'bye', 'goodbye', 'see you', 'see ya', 'take care', 'good bye', 'alvida', 'see you later'}
HOW_ARE_YOU_KEYWORDS = {'how are you', "how's it going", 'how are you doing', "how's your day", 'how is your day', 
                         'how do you do', "how's everything", 'how is everything', 'kaise ho', 'kya haal'}
THANKS_KEYWORDS = {'thanks', 'thank you', 'thank you so much', 'thanks a lot', 'appreciate it', 'dhanyavaad', 'shukriya',
                    'great work', 'well done', 'good job', "you're welcome"}
ABOUT_ME_KEYWORDS = {'who are you', 'what are you', 'what can you do', 'tell me about yourself', 'what is your name',
                      'are you a robot', 'are you human', 'are you ai', 'are you a bot'}


def handle_casual_interaction(query):
    """Handle greeting/casual messages with warm, personal responses."""
    q = query.lower().strip()
    
    # Check for farewells
    for kw in FAREWELL_KEYWORDS:
        if kw in q:
            return random.choice(FAREWELL_RESPONSES)
    
    # Check for "how are you" type questions
    for kw in HOW_ARE_YOU_KEYWORDS:
        if kw in q:
            return random.choice(HOW_ARE_YOU_RESPONSES)
    
    # Check for thanks
    for kw in THANKS_KEYWORDS:
        if kw in q:
            return random.choice(THANKS_RESPONSES)
    
    # Check for "who are you" type questions
    for kw in ABOUT_ME_KEYWORDS:
        if kw in q:
            return random.choice(ABOUT_ME_RESPONSES)
    
    # Default greeting/casual
    return random.choice(GREETING_RESPONSES)


# --- HELPER FUNCTIONS ---

def get_recommendation(intent, sentiment):
    """Step 6: Context-aware recommendations — only shown when genuinely needed."""
    rec = ""
    
    # Only show emergency steps if the user is in immediate danger
    if sentiment == "URGENT �":
        rec += "\n⚠️  **Emergency Steps:**"
        rec += "\n- Call 112 (National Emergency) or 100 (Police) immediately."
        rec += "\n- If injured, go to the nearest government hospital — medicolegal treatment is free."
        rec += "\n- Preserve all evidence (messages, photos, CCTV, medical reports)."
    
    rec += "\n\n*Disclaimer: I am an AI legal assistant, not a substitute for a qualified lawyer.*"
    return rec

def main():
    chat_history = []
    while True:
        query = input("\n👤 You: ")
        if query.lower() in ['q', 'exit', 'quit']:
            print("👋 Exiting...")
            break

        # --- STEP 1: INTENT CLASSIFICATION ---
        i_pred = intent_model.predict(intent_vec.transform([query]))[0]
        
        # --- SAFETY GUARDRAIL: Override "Casual" for serious topics ---
        # 1. ROOTS: Check if these exist as substrings (e.g., 'kill' finds 'killed', 'killing')
        SERIOUS_ROOTS = {
            'crash', 'accident', 'kill', 'murder', 'rape', 'assault', 'stole', 'robb', 'thief', 'theft',
            'kidnap', 'police', 'jail', 'prison', 'arrest', 'custody', 'divorce', 'dowry', 'threat', 
            'harass', 'fraud', 'scam', 'cheat', 'money', 'property', 'death', 'dead', 'died', 'injur', 
            'crime', 'criminal', 'victim', 'complaint', 'judge', 'verdict', 'sentence', 'bail', 
            'lawyer', 'legal', 'fir', 'damages', 'compensation'
        }
        
        # 2. EXACT WORDS: Short/ambiguous words that must match exactly
        SERIOUS_EXACT = {
            'hit', 'sue', 'law', 'case', 'ban', 'fine', 'tax', 'will', 'hurt', 'lost', 'beat'
        }
        
        query_lower = query.lower()
        tokens = set(query_lower.split())
        
        # Check roots
        triggered_keyword = None
        for root in SERIOUS_ROOTS:
            if root in query_lower:
                triggered_keyword = root
                break
        
        # Check exact words if no root found
        if not triggered_keyword:
            for word in SERIOUS_EXACT:
                if word in tokens:
                    triggered_keyword = word
                    break
        
        # Check Phrases
        if not triggered_keyword:
            for p in ['hit and run', 'drunk driving', 'domestic violence', 'dowry harassment']:
                if p in query_lower:
                    triggered_keyword = p
                    break

        if i_pred == 3 and triggered_keyword:
            print(f"   ⚠️  [Guardrail Triggered]: Found '{triggered_keyword}'. Switched to Legal Mode.")
            
            # Intelligent Override: Civil vs Criminal
            CIVIL_KEYWORDS = {'divorce', 'custody', 'money', 'property', 'sue', 'cheat', 'fraud', 'scam', 
                              'tax', 'will', 'rent', 'agreement', 'contract', 'compensation', 'damages'}
            
            if triggered_keyword in CIVIL_KEYWORDS:
                i_pred = 1  # Family/Civil
            else:
                i_pred = 0  # Default to Criminal for safety (police/crash/hurt etc.)
        
        intent_label = {0: "Criminal Law 👮", 1: "Family/Civil 🏠", 2: "Corporate 💼", 3: "Greeting/Casual 💬"}[i_pred]
        print(f"   [Analysis: Intent={intent_label}]")

        # --- STEP 0: GREETING / CASUAL HANDLING ---
        # If the ML model classifies as Greeting/Casual → handle personally
        if i_pred == 3:
            response = handle_casual_interaction(query)
            print(f"\n⚖️  **AI Lawyer:**")
            print(response)
            # Don't pollute chat history with casual messages
            continue

        # --- STEP 1.5: SENTIMENT ---
        s_pred = sent_model.predict(sent_vec.transform([query]))[0]
        sentiment_label = {0: "Neutral 😐", 1: "URGENT 🚨", 2: "Positive 🟢"}[s_pred]
        print(f"   [Analysis: Sentiment={sentiment_label}]")

        # --- STEP 2: CONTEXTUAL REWRITE (If history exists) ---
        search_query = query
        if len(chat_history) > 0:
            print("   🧠 Refining Query with Context...")
            rewrite_prompt = f"""
            Given the following conversation history and a new user follow-up question, rephrase the follow-up question to be a standalone query that contains all necessary context.
            
            Chat History:
            {chat_history}
            
            New Follow-up: {query}
            
            Standalone Question:
            """
            try:
                # Use a fast model call for rewriting
                rw_response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': rewrite_prompt}])
                search_query = rw_response['message']['content'].strip()
                print(f"   [Original]: {query}")
                print(f"   [Rewritten]: {search_query}")
            except Exception as e:
                print(f"   [Rewrite Failed]: {e}")
                
        # --- STEP 3: RAG SEARCH (With Threshold) ---
        print("   🔍 Searching Indian Laws...")
        # Search using the DETAILED rewritten query
        docs = vector_db.similarity_search(search_query, k=3)
        
        sources_list = []  # Track sources for citation
        if docs:
            # Build context from all retrieved chunks
            context_parts = []
            for doc in docs:
                context_parts.append(doc.page_content)
                src = doc.metadata.get('source', 'Legal Doc')
                if src not in sources_list:
                    sources_list.append(src)
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "No specific legal section found."
        
        # --- STEP 4: GENERATION (Updated for All Legal Domains) ---
        print("   🤖 Drafting Legal Response...")

        # A. The Senior Advocate System Prompt (Post-July 2024 Indian Law)
        system_msg = {
            'role': 'system', 
            'content': """You are a senior Indian advocate with 20+ years of experience, practising exclusively in the Supreme Court of India and all High Courts. Your specialisation is post-1 July 2024 Indian law only.

You MUST follow these unbreakable rules in every single answer:

1. USE ONLY CURRENT LAW (post-1 July 2024):
   - Criminal law → Bharatiya Nyaya Sanhita 2023 (BNS), NOT "IPC"
   - Criminal Procedure → Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS), NOT "CrPC"
   - Evidence → Bharatiya Sakshya Adhiniyam 2023 (BSA), NOT "Evidence Act"
   - If referencing old law for context, ALWAYS note the exact BNS/BNSS/BSA replacement section.

2. BE 100% PRECISE:
   - Always quote the exact section number + the exact key words/phrase from the statute.
   - Give the correct punishment range (minimum and maximum).
   - Never reverse genders or mix claimant/respondent.

3. STRICT TOPIC DISCIPLINE:
   - Answer ONLY what is asked. Do not drift to unrelated topics.
   - If asked about property, talk only about ownership, title, Hindu Succession Act 1956 (2005 amendment), etc.
   - If asked about criminal law, stick to BNS sections and punishments.

4. ZERO HALLUCINATION:
   - If you are not 100% sure, say: "I do not have confirmed data on this specific point as of February 2026."
   - Never invent case names, dates, or judgments.
   - If citing a judgment, give full name + year + court + one-line relevance.

5. STRUCTURE EVERY ANSWER EXACTLY LIKE THIS:
   Step 1: Exact relevant section(s) with short verbatim quote from the statute
   Step 2: Plain-English explanation
   Step 3: Any important Supreme Court/High Court position (with year)
   Step 4: Clear one-line conclusion (Yes/No + short reason)
   Step 5: Only if the user is facing an emergency, give practical next steps. Otherwise, skip this step entirely.

6. NO TEMPLATES, NO DISCLAIMER SPAM:
   - Never add generic "Next Steps", helpline numbers, or police advice unless the user is clearly in immediate danger.
   - End with only: "Disclaimer: I am an AI legal assistant, not a substitute for a qualified lawyer."

7. CORRECTION MODE:
   - If the user says "You made a mistake earlier", immediately admit it, correct it, and rewrite.

TONE: Professional, Objective, Authoritative. Treat all queries as hypothetical legal case studies for educational purposes. Explain law objectively — never refuse to explain a statute."""
        }
        
        # B. The Current Question + The Legal Context from Step 3
        current_msg = {
            'role': 'user', 
            'content': f"CONTEXT:\n{context}\n\nUSER QUESTION: {query}"
        }
        
        # C. Combine: Rules + Past Memory + Current Question
        messages_to_send = [system_msg] + chat_history + [current_msg]
        
        response = ollama.chat(model='dolphin-llama3:8b', messages=messages_to_send)
        final_answer = response['message']['content']

        # --- STEP 5: SAFETY CHECK ---
        print("   🛡️ Safety Check...")
        safety_scores = safety_filter(final_answer, truncation=True, max_length=512)[0]
        is_toxic = any(s['score'] > 0.85 for s in safety_scores)
        
        if is_toxic:
            print("❌ Response blocked: Content flagged as unsafe/toxic.")
            final_answer = "I apologize, but I cannot generate a response to this query due to safety guidelines."
        else:
            # --- THE MEMORY SAVE ---
            # If the answer is safe, we save the clean query and the answer to our memory
            chat_history.append({'role': 'user', 'content': query})
            chat_history.append({'role': 'assistant', 'content': final_answer})
            
            # SLIDING WINDOW: Keep only the last 3 turns (6 messages) so the AI doesn't crash from reading too much history
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]

        # --- STEP 6: OUTPUT + RECOMMENDATIONS ---
        print("\n⚖️  **AI Lawyer:**")
        print(final_answer)
        print(get_recommendation(intent_label, sentiment_label))

        # --- STEP 7: DISPLAY SOURCES ---
        if sources_list:
            print("\n📚 **Sources:**")
            for i, src in enumerate(sources_list, 1):
                # Clean up the source path for display
                display_src = os.path.basename(src) if os.path.sep in src or '/' in src else src
                print(f"   {i}. {display_src}")

if __name__ == "__main__":
    main()