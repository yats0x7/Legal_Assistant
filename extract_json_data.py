"""
extract_json_data.py
Extracts data from the two new JSON files and generates enriched training CSVs
for intent classification and sentiment analysis.
"""
import json
import os
import re
import random
import pandas as pd

DATA_DIR = "data"

# --- KEYWORD DICTIONARIES FOR INTENT CLASSIFICATION ---

CRIMINAL_KEYWORDS = [
    'murder', 'homicide', 'culpable', 'kill', 'death', 'manslaughter',
    'theft', 'robbery', 'dacoity', 'extortion', 'burglary', 'stealing', 'loot',
    'assault', 'battery', 'grievous hurt', 'voluntarily causing hurt', 'attack',
    'kidnapping', 'abduction', 'wrongful confinement', 'wrongful restraint',
    'rape', 'sexual assault', 'molestation', 'outraging modesty', 'sexual harassment',
    'forgery', 'counterfeiting', 'fabricating evidence', 'perjury',
    'criminal conspiracy', 'criminal intimidation', 'criminal breach of trust',
    'cheating', 'fraud', 'misappropriation', 'dishonest',
    'defamation', 'obscene', 'public nuisance',
    'rioting', 'unlawful assembly', 'affray', 'sedition', 'waging war',
    'fir', 'police', 'arrest', 'bail', 'cognizable', 'non-bailable',
    'warrant', 'custody', 'remand', 'charge sheet', 'investigation',
    'ipc', 'bns', 'crpc', 'bnss', 'indian penal code', 'bharatiya nyaya',
    'penal', 'punishment', 'imprisonment', 'fine', 'death sentence',
    'accused', 'convict', 'prosecution', 'complainant', 'victim',
    'abetment', 'attempt', 'criminal', 'offence', 'offense',
    'dowry death', 'cruelty', 'domestic violence',
    'narcotics', 'ndps', 'drugs', 'smuggling', 'trafficking',
    'cybercrime', 'hacking', 'identity theft',
    'terrorism', 'uapa', 'national security',
    'arms act', 'explosive', 'weapon',
    'corruption', 'bribery', 'prevention of corruption',
    'sc/st act', 'atrocity', 'scheduled caste', 'scheduled tribe',
    'pocso', 'child abuse', 'juvenile',
    'anticipatory bail', 'regular bail', 'default bail',
    'section 302', 'section 304', 'section 307', 'section 376',
    'section 420', 'section 498', 'section 354', 'section 506',
    'section 34', 'section 149', 'section 120b',
]

CIVIL_KEYWORDS = [
    'divorce', 'annulment', 'judicial separation', 'restitution of conjugal',
    'alimony', 'maintenance', 'streedhan', 'mehr',
    'child custody', 'guardianship', 'visitation', 'adoption',
    'marriage', 'nikah', 'hindu marriage', 'special marriage', 'court marriage',
    'property', 'land', 'title', 'possession', 'partition', 'inheritance',
    'ancestral property', 'succession', 'will', 'testament', 'probate',
    'tenancy', 'rent', 'landlord', 'tenant', 'eviction', 'lease',
    'specific relief', 'injunction', 'declaration', 'specific performance',
    'tort', 'negligence', 'nuisance', 'trespass', 'defamation civil',
    'consumer complaint', 'consumer protection', 'consumer forum',
    'family court', 'civil court', 'civil suit', 'civil case',
    'limitation act', 'limitation period', 'time barred',
    'transfer of property', 'sale deed', 'gift deed', 'mortgage',
    'registration act', 'stamp duty', 'power of attorney',
    'hindu succession', 'muslim personal law', 'christian law',
    'domestic violence act', 'protection of women',
    'rti', 'right to information',
    'writ petition', 'fundamental rights', 'article 14', 'article 19',
    'article 21', 'constitution', 'constitutional', 'directive principles',
    'preamble', 'amendment', 'parliament', 'legislature',
    'supreme court', 'high court', 'district court',
    'citizenship', 'election', 'panchayat', 'municipality',
    'union', 'state', 'territory', 'federal',
]

CORPORATE_KEYWORDS = [
    'gst', 'goods and services tax', 'income tax', 'corporate tax',
    'tax evasion', 'tax return', 'tax filing', 'assessment',
    'company', 'incorporation', 'memorandum', 'articles of association',
    'board of directors', 'shareholders', 'share capital', 'debenture',
    'startup', 'llp', 'partnership', 'sole proprietorship',
    'contract', 'agreement', 'indemnity', 'guarantee', 'consideration',
    'breach of contract', 'damages', 'penalty clause',
    'cheque bounce', 'negotiable instrument', 'promissory note',
    'insolvency', 'bankruptcy', 'ibc', 'nclt', 'nclat',
    'merger', 'acquisition', 'amalgamation', 'demerger',
    'intellectual property', 'patent', 'trademark', 'copyright',
    'employment', 'labour', 'labor', 'industrial dispute', 'termination',
    'salary', 'wages', 'provident fund', 'gratuity', 'esi',
    'sebi', 'securities', 'stock market', 'insider trading',
    'competition act', 'monopoly', 'anti-competitive',
    'arbitration', 'mediation', 'adr', 'conciliation',
    'information technology act', 'it act', 'cyber', 'data protection',
    'foreign exchange', 'fema', 'fdi', 'foreign investment',
    'rbi', 'banking', 'nbfc', 'financial',
    'companies act', 'msme', 'udyam',
    'environmental law', 'pollution', 'ngt', 'environmental clearance',
    'real estate regulation', 'rera', 'builder', 'developer',
]

URGENT_KEYWORDS = [
    'urgent', 'emergency', 'help', 'immediately', 'right now', 'please help',
    'threatening', 'danger', 'attack', 'beaten', 'beating', 'kidnapped',
    'suicide', 'life threat', 'hostage', 'trapped', 'bomb', 'weapon',
    'molested', 'raped', 'stabbed', 'shot', 'bleeding',
    'missing', 'abducted', 'detained', 'arrested without',
    'breaking into', 'followed', 'stalked', 'harassed',
]

POSITIVE_KEYWORDS = [
    'won', 'victory', 'favor', 'favour', 'successful', 'granted',
    'dismissed', 'quashed', 'acquitted', 'discharged', 'settled',
    'resolved', 'compensation', 'refund', 'recovered', 'cleared',
    'happy', 'relieved', 'grateful', 'thankful', 'satisfied',
    'thank you', 'thanks', 'appreciate', 'well done', 'good job',
]


def classify_intent(text):
    """Classify a text into intent category: 0=Criminal, 1=Civil, 2=Corporate, 3=Casual"""
    text_lower = text.lower()

    scores = {0: 0, 1: 0, 2: 0}

    for kw in CRIMINAL_KEYWORDS:
        if kw in text_lower:
            scores[0] += 1

    for kw in CIVIL_KEYWORDS:
        if kw in text_lower:
            scores[1] += 1

    for kw in CORPORATE_KEYWORDS:
        if kw in text_lower:
            scores[2] += 1

    max_score = max(scores.values())
    if max_score == 0:
        # Default: if no keywords match, try to guess from common patterns
        if any(w in text_lower for w in ['section', 'act', 'law', 'legal', 'court', 'judge']):
            return 1  # Default to Civil for generic legal queries
        return 0  # Default to Criminal (most common in Indian legal Q&A)

    # Return the intent with highest score
    return max(scores, key=scores.get)


def classify_sentiment(text):
    """Classify sentiment: 0=Neutral, 1=Urgent, 2=Positive"""
    text_lower = text.lower()

    urgent_score = sum(1 for kw in URGENT_KEYWORDS if kw in text_lower)
    positive_score = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)

    if urgent_score > positive_score and urgent_score > 0:
        return 1
    elif positive_score > urgent_score and positive_score > 0:
        return 2
    return 0


def load_indic_legal_qa():
    """Load IndicLegalQA Dataset_10K.json"""
    filepath = os.path.join(DATA_DIR, "IndicLegalQA Dataset_10K.json")
    if not os.path.exists(filepath):
        print(f"   ⚠️  File not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"   📄 Loaded {len(data)} entries from IndicLegalQA Dataset")
    return data


def load_constitution_qa():
    """Load constitution_qa.json"""
    filepath = os.path.join(DATA_DIR, "constitution_qa.json")
    if not os.path.exists(filepath):
        print(f"   ⚠️  File not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"   📄 Loaded {len(data)} entries from Constitution Q&A")
    return data


def generate_intent_data():
    """Generate enriched intent_data.csv from JSON files + templates"""
    print("\n📊 Generating Intent Training Data...")

    intent_data = []

    # 1. Extract from IndicLegalQA
    indic_data = load_indic_legal_qa()
    for item in indic_data:
        question = item.get('question', '')
        answer = item.get('answer', '')
        if question:
            combined = f"{question} {answer}"
            intent = classify_intent(combined)
            intent_data.append({'text': question, 'intent': intent})

    # 2. Extract from Constitution Q&A (mostly Civil/Constitutional → intent 1)
    const_data = load_constitution_qa()
    for item in const_data:
        question = item.get('question', '')
        if question:
            # Constitution questions are predominantly civil/constitutional
            intent = classify_intent(question)
            # Give a slight bias toward Civil for constitution questions
            if intent == 0 and not any(kw in question.lower() for kw in ['murder', 'theft', 'rape', 'assault', 'fir', 'arrest', 'bail']):
                intent = 1  # Constitutional → Civil
            intent_data.append({'text': question, 'intent': intent})

    # 3. Add template-generated data for balance
    criminal_kws = ["arrest", "bail", "police", "FIR", "jail", "murder", "theft", "assault", "warrant", "custody", "crime"]
    family_kws = ["divorce", "alimony", "child custody", "property", "rent", "landlord", "tenant", "will", "ancestral", "marriage"]
    corporate_kws = ["GST", "tax", "company", "startup", "incorporation", "cheque bounce", "contract", "agreement", "salary", "employment"]

    templates = [
        "How to file a case for {keyword}?",
        "What is the punishment for {keyword} in India?",
        "My neighbor is involved in {keyword}.",
        "Is {keyword} legal in Delhi?",
        "Procedure to apply for {keyword}.",
        "Can I get a lawyer for {keyword}?",
        "Latest supreme court judgment on {keyword}.",
        "Rules regarding {keyword} 2024.",
        "I am facing an issue with {keyword}.",
        "Help me with {keyword} laws.",
        "What are the rights in case of {keyword}?",
        "Explain the law about {keyword}.",
        "What happens if someone does {keyword}?",
        "Is there any remedy for {keyword}?",
        "What is the procedure after {keyword}?",
    ]

    for _ in range(150):
        text = random.choice(templates).format(keyword=random.choice(criminal_kws))
        intent_data.append({'text': text, 'intent': 0})

    for _ in range(150):
        text = random.choice(templates).format(keyword=random.choice(family_kws))
        intent_data.append({'text': text, 'intent': 1})

    for _ in range(150):
        text = random.choice(templates).format(keyword=random.choice(corporate_kws))
        intent_data.append({'text': text, 'intent': 2})

    # 4. Add Greeting/Casual examples (intent 3) - these need manual examples
    casual_examples = [
        "hello", "hi", "hey", "hey there", "good morning",
        "good afternoon", "good evening", "namaste", "what's up",
        "how are you", "how's it going", "how are you doing",
        "bye", "goodbye", "see you later", "take care",
        "thank you", "thanks", "thanks a lot", "appreciate it",
        "who are you", "what can you do", "what is your name",
        "are you a robot", "are you human", "tell me about yourself",
        "hi there, how are you doing today",
        "good morning, hope you're doing well",
        "hey, nice to meet you",
        "hello, I just wanted to say hi",
        "what's your name?",
        "can you help me",
        "I need some help",
        "nice talking to you",
        "that's interesting",
        "okay cool",
        "alright then",
        "sounds good",
        "I see",
        "got it",
        "hmm okay",
        "hello ji", "namaskar", "kaise ho",
        "kya haal hai", "shukriya", "dhanyavaad",
        "acha theek hai", "bye bye", "alvida",
        "good night", "hola", "yo", "sup",
        "hi bot", "hello AI", "greetings",
        "howdy", "what's happening", "how do you do",
    ]

    for text in casual_examples:
        intent_data.append({'text': text, 'intent': 3})

    # Duplicate casual with slight variations to balance class
    for _ in range(100):
        base = random.choice(casual_examples)
        variations = [base, base + "!", base + "?", base.capitalize(), base.upper()]
        intent_data.append({'text': random.choice(variations), 'intent': 3})

    # Create DataFrame, shuffle, and save
    df = pd.DataFrame(intent_data)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    csv_path = os.path.join(DATA_DIR, 'intent_data.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n   ✅ Intent data saved to {csv_path}")
    print(f"   📈 Total samples: {len(df)}")
    print(f"   📊 Class distribution:")
    label_names = {0: "Criminal", 1: "Family/Civil", 2: "Corporate", 3: "Greeting/Casual"}
    for intent_id, count in df['intent'].value_counts().sort_index().items():
        print(f"      {label_names[intent_id]}: {count}")

    return df


def generate_sentiment_data():
    """Generate enriched sentiment_data.csv from JSON files + existing data"""
    print("\n📊 Generating Sentiment Training Data...")

    sentiment_data = []

    # 1. Load existing sentiment data
    existing_path = os.path.join(DATA_DIR, 'sentiment_data.csv')
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)
        if len(existing_df) > 0 and 'text' in existing_df.columns and 'label' in existing_df.columns:
            for _, row in existing_df.iterrows():
                sentiment_data.append({'text': row['text'], 'label': int(row['label'])})
            print(f"   📄 Loaded {len(existing_df)} existing sentiment examples")

    # 2. Extract from IndicLegalQA - classify Q&A pairs by sentiment
    indic_data = load_indic_legal_qa()
    sampled = random.sample(indic_data, min(500, len(indic_data)))
    for item in sampled:
        question = item.get('question', '')
        if question:
            sentiment = classify_sentiment(question)
            sentiment_data.append({'text': question, 'label': sentiment})

    # 3. Add more template-generated examples for balance

    # Neutral examples (informational questions)
    neutral_templates = [
        "What is the procedure for {topic}?",
        "How does {topic} work in India?",
        "Can you explain {topic} under Indian law?",
        "What are the rules regarding {topic}?",
        "Is {topic} applicable in my case?",
        "What documents are needed for {topic}?",
        "How long does {topic} take?",
        "What is the fee for {topic}?",
        "Where do I file for {topic}?",
        "Who is eligible for {topic}?",
    ]
    neutral_topics = [
        "filing FIR", "divorce petition", "bail application", "RTI",
        "property registration", "trademark filing", "GST return",
        "consumer complaint", "legal notice", "power of attorney",
        "court marriage", "succession certificate", "writ petition",
        "appeal in high court", "company incorporation", "patent filing",
    ]
    for _ in range(150):
        text = random.choice(neutral_templates).format(topic=random.choice(neutral_topics))
        sentiment_data.append({'text': text, 'label': 0})

    # Urgent examples
    urgent_examples = [
        "Someone is trying to kill me please help!",
        "I am being beaten right now, call police!",
        "My daughter has been kidnapped, urgent help needed!",
        "I am going to commit suicide because of harassment.",
        "They are threatening to acid attack me!",
        "Please help, someone is breaking into my house RIGHT NOW!",
        "My child is missing since yesterday, please help!",
        "I am being held hostage, help me!",
        "My wife is being beaten by in-laws, call police now!",
        "There is a bomb threat in our building, emergency!",
        "Someone put a gun to my head and took my money!",
        "I am being followed by unknown people, help!",
        "My sister was molested today, we need urgent help!",
        "Domestic violence happening right now, send police!",
        "I was stabbed, please help me!",
        "Life threatening situation, help immediately!",
        "Someone spiked my drink, I feel unsafe!",
        "Recovery agents are breaking my door right now!",
        "My father is having a medical emergency due to police beating!",
        "I am locked inside and they won't let me leave!",
        "My family is being attacked by a mob right now!",
        "Police are torturing my brother in custody!",
        "They kidnapped my son and demanding ransom!",
        "I am being blackmailed with private photos!",
        "Someone is trying to poison my food!",
        "A gang is looting our neighborhood right now!",
        "My pregnant wife is being beaten, urgent help!",
        "Armed men have entered our village!",
        "I am trapped and cannot escape this place!",
        "They are forcing me into marriage against my will!",
    ]
    for text in urgent_examples:
        sentiment_data.append({'text': text, 'label': 1})

    # Positive examples
    positive_examples = [
        "We got the favorable judgment finally!",
        "My case was decided in my favor today!",
        "The judge granted me bail, I am so happy!",
        "My property dispute was finally resolved!",
        "I received compensation for wrongful termination!",
        "The mediation was successful, we reached agreement!",
        "My divorce was granted peacefully!",
        "The insurance company finally paid my claim!",
        "I got my passport back after the court order!",
        "The harassment has stopped after filing the complaint!",
        "My employer agreed to pay all my dues!",
        "The false case against me was quashed today!",
        "Finally got custody of my children!",
        "I am grateful for the legal aid provided!",
        "The police finally registered my FIR!",
        "My stolen property was recovered by police!",
        "The tenant finally vacated after court order!",
        "My name was cleared of all charges!",
        "The appeal was successful and sentence was reduced!",
        "I won the consumer case and got full refund!",
        "Thank you so much for the amazing guidance!",
        "Your advice helped me win the case!",
        "Court ruled in my favor, justice is served!",
        "Got the stay order, feeling relieved!",
        "The settlement was very favorable for us!",
        "My bail was granted immediately, so grateful!",
        "The landlord returned my security deposit finally!",
        "All charges were dropped, I am free!",
        "The arbitration award was in my favor!",
        "My pension was finally approved after the complaint!",
    ]
    for text in positive_examples:
        sentiment_data.append({'text': text, 'label': 2})

    # Create DataFrame, shuffle, and save
    df = pd.DataFrame(sentiment_data)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    csv_path = os.path.join(DATA_DIR, 'sentiment_data.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n   ✅ Sentiment data saved to {csv_path}")
    print(f"   📈 Total samples: {len(df)}")
    print(f"   📊 Class distribution:")
    label_names = {0: "Neutral", 1: "Urgent", 2: "Positive"}
    for label_id, count in df['label'].value_counts().sort_index().items():
        print(f"      {label_names[label_id]}: {count}")

    return df


if __name__ == "__main__":
    print("--- 📦 EXTRACTING DATA FROM JSON FILES ---\n")

    intent_df = generate_intent_data()
    sentiment_df = generate_sentiment_data()

    print("\n--- ✅ DATA EXTRACTION COMPLETE ---")
    print(f"   Intent samples: {len(intent_df)}")
    print(f"   Sentiment samples: {len(sentiment_df)}")
