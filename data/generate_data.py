# import pandas as pd
# import os

# # Create folder if missing
# if not os.path.exists('data'):
#     os.makedirs('data')

# # Labels:
# # 0 = Neutral (Information/Procedure)
# # 1 = Urgent/Crisis (Danger/Panic)
# # 2 = Positive (Success/Gratitude)

# data = {
#     'text': [
#         # --- Neutral (Information/Procedure) ---
#         "What is the procedure to file GST returns?",
#         "How much does a affidavit cost?",
#         "Is digital signature valid in court?",
#         "What are the documents for marriage registration?",
#         "Can I file an RTI online?",
#         "Difference between Cheque Bounce and Fraud?",
#         "What is the time limit for filing an appeal?",
#         "Does a will need to be registered?",
#         "How to apply for a panic button on mobile?",
#         "Is there a ladies special court in Delhi?",
#         "What is section 144?",
#         "Explain the process of plea bargaining.",
#         "How to check case status online?",
#         "Is dowry demand a bailable offense?",
#         "What is the minimum wage in Haryana?",

#         # --- Urgent/Crisis (High Priority) ---
#         "Police are beating my brother in custody right now!",
#         "My husband is threatening to kill me and my kids.",
#         "Recovery agents are banging on my door!",
#         "I have been falsely accused of rape, please help.",
#         "Emergency! My landlord threw my things on the street.",
#         "My business partner ran away with all the money.",
#         "I am being blackmailed with my private photos.",
#         "Suicide threat due to loan app harassment.",
#         "They are denying me bail, I am scared.",
#         "My neighbor is attacking my family with a weapon.",
#         "Domestic violence happening now, need police.",
#         "Kidnapping threat received on phone.",
#         "Police refusing to file FIR for missing child.",
#         "I am trapped in a fraudulent marriage.",
#         "Sexual harassment at workplace, boss is forcing me.",

#         # --- Positive (Success/Gratitude) ---
#         "Thank you so much, the advice really helped!",
#         "We won the case today in the High Court!",
#         "Finally got my divorce decree, I am so relieved.",
#         "The court granted me bail! Thank you.",
#         "The judge dismissed the fake case against me.",
#         "Received the full compensation amount today.",
#         "Problem resolved after sending the legal notice.",
#         "Great guidance, you saved me a lot of trouble.",
#         "I am very happy with the resolution.",
#         "The police finally filed the FIR, thanks to you.",
#         "Appreciate your quick help, it worked.",
#         "Justice prevailed! The culprit is arrested.",
#         "My property dispute is settled peacefully.",
#         "Got my refund from the builder.",
#         "Excellent support, I am free now."
#     ],
#     'label': [
#         # 15 Neutral
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         # 15 Urgent
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         # 15 Positive
#         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
#     ]
# }

# df = pd.DataFrame(data)
# csv_path = os.path.join('data', 'sentiment_data.csv')
# df.to_csv(csv_path, index=False)

# print(f"✅ Generated {len(df)} training examples at: {csv_path}")








import pandas as pd
import os
import random

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# --- 1. Define Templates for Variety ---

# NEUTRAL (Information, Procedure, Laws) - Label 0
neutral_templates = [
    "What is the procedure for {topic}?",
    "How to file a {topic} in India?",
    "Documents required for {topic} application.",
    "Is {topic} legal in India?",
    "Can I apply for {topic} online?",
    "What is the limitation period for {topic}?",
    "Difference between {topic} and {topic2}?",
    "Explain section {sec} of {act}.",
    "Is there a format for {topic}?",
    "Fees for registering a {topic}."
]
neutral_topics = [
    "GST return", "divorce", "marriage registration", "RTI", "trademark", 
    "copyright", "patent", "will", "gift deed", "power of attorney", 
    "sale deed", "rent agreement", "consumer complaint", "FIR", "bail"
]
neutral_acts = ["IPC", "CrPC", "BNS", "Contract Act", "IT Act", "Hindu Marriage Act"]

# URGENT (Crisis, Threat, Panic) - Label 1
urgent_templates = [
    "Police are {action} my {person} right now!",
    "My {person} is threatening to {threat} me.",
    "I have been {crime} by my {person}.",
    "Emergency! {entity} is {action} my house.",
    "They are {action} me for money I didn't take.",
    "Help! I am being {crime} and don't know what to do.",
    "My {person} has been detained without a warrant.",
    "Suicide threat because of {entity} harassment.",
    "I am trapped in a {situation} and need help.",
    "False accusation of {crime} against me."
]
urgent_actions = ["beating", "arresting", "harassing", "blackmailing", "evicting", "attacking"]
urgent_persons = ["husband", "wife", "neighbor", "boss", "landlord", "partner", "brother"]
urgent_crimes = ["raped", "assaulted", "cheated", "robbed", "abused", "stalked"]
urgent_entities = ["bank agents", "police", "goons", "loan sharks", "in-laws"]

# POSITIVE (Success, Gratitude, Relief) - Label 2
positive_templates = [
    "Thank you for the {adj} advice.",
    "We finally {outcome} the case!",
    "The court {outcome} in my favor.",
    "I am {feeling} because the issue is resolved.",
    "Got my {noun} back from the {person}.",
    "Appreciate your {adj} help with the {topic}.",
    "The legal notice worked and they {outcome}.",
    "Judge gave a {adj} verdict today.",
    "My {topic} was approved successfully.",
    "Justice prevailed, I am {feeling} now."
]
positive_outcomes = ["won", "settled", "dismissed", "closed", "cleared"]
positive_adj = ["excellent", "great", "quick", "superb", "helpful", "fair"]
positive_feelings = ["relieved", "happy", "safe", "free", "satisfied"]

# --- 2. Generate Data ---
data = []

# Generate 100 Neutral Examples
for _ in range(100):
    t = random.choice(neutral_templates)
    text = t.format(
        topic=random.choice(neutral_topics),
        topic2=random.choice(neutral_topics),
        sec=random.randint(1, 500),
        act=random.choice(neutral_acts)
    )
    data.append([text, 0])

# Generate 100 Urgent Examples
for _ in range(100):
    t = random.choice(urgent_templates)
    text = t.format(
        action=random.choice(urgent_actions),
        person=random.choice(urgent_persons),
        threat="kill", 
        crime=random.choice(urgent_crimes),
        entity=random.choice(urgent_entities),
        situation="dangerous situation"
    )
    data.append([text, 1])

# Generate 100 Positive Examples
for _ in range(100):
    t = random.choice(positive_templates)
    text = t.format(
        adj=random.choice(positive_adj),
        outcome=random.choice(positive_outcomes),
        feeling=random.choice(positive_feelings),
        noun="money",
        person="builder",
        topic="refund"
    )
    data.append([text, 2])

# --- 3. Save to CSV ---
df = pd.DataFrame(data, columns=['text', 'label'])
# Shuffle the data so the model doesn't learn order
df = df.sample(frac=1).reset_index(drop=True)

csv_path = os.path.join('data', 'sentiment_data.csv')
df.to_csv(csv_path, index=False)

print(f"✅ Generated {len(df)} training examples at: {csv_path}")
print("   - 100 Neutral")
print("   - 100 Urgent")
print("   - 100 Positive")