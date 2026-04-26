from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import json, os, math, string, httpx, re
from collections import Counter
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NLP QA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWLEDGE_BASE_PATH = os.getenv("KB_PATH", "knowledge_base.json")

def load_knowledge_base():
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

knowledge_base = load_knowledge_base()

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and",
    "or","but","not","with","this","that","are","was","were","be",
    "been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","i","you",
    "he","she","we","they","what","how","why","when","where","which",
    "who","whom","whose","about","from","by","as","so","if","then",
}

# ── Built-in dictionary for common everyday words ─────────────────────────────
BUILTIN_ANSWERS = {
    "chair": "A chair is a piece of furniture designed for a single person to sit on, typically having four legs and a back. Chairs are used in homes, offices, schools, and many other settings.",
    "table": "A table is a piece of furniture with a flat top surface supported by legs, used for placing objects on or working at. Tables are used for eating, writing, and many other activities.",
    "computer": "A computer is an electronic device that processes data and performs calculations according to programmed instructions. Modern computers can perform billions of operations per second and are used for work, communication, and entertainment.",
    "phone": "A phone is a communication device that allows people to talk to each other over long distances. Modern smartphones also function as computers, cameras, and internet devices.",
    "book": "A book is a written or printed work consisting of pages bound together, containing text, images, or both. Books are used for reading, education, and entertainment.",
    "car": "A car is a wheeled motor vehicle used for transportation, typically powered by a petrol or electric engine. Cars are one of the most common forms of personal transportation worldwide.",
    "house": "A house is a building used as a home or residence for people. Houses provide shelter and living space and can vary greatly in size and design.",
    "water": "Water is a transparent, tasteless, odourless liquid that is essential for all known forms of life. It covers about 71 percent of the Earth's surface and is vital for drinking, agriculture, and industry.",
    "food": "Food is any nutritious substance that organisms consume to obtain energy and support growth. It includes plants, animals, and fungi and is essential for survival.",
    "tree": "A tree is a large woody plant with a trunk, branches, and leaves that grows to a significant height. Trees produce oxygen, provide shade, and are home to many animals.",
    "sun": "The Sun is the star at the center of our solar system that provides light and heat to Earth. It is about 93 million miles from Earth and is essential for life on our planet.",
    "moon": "The Moon is Earth's only natural satellite, orbiting our planet at an average distance of 384,400 kilometers. It influences tides and is the brightest object in the night sky.",
    "sky": "The sky is the expanse of atmosphere and space visible from Earth's surface. It appears blue during the day due to the scattering of sunlight and dark at night revealing stars.",
    "rain": "Rain is liquid water that falls from clouds in the atmosphere to the Earth's surface. It is part of the water cycle and is essential for replenishing freshwater supplies.",
    "fire": "Fire is the rapid oxidation of material producing heat, light, and various gases. It has been used by humans for cooking, warmth, and light for hundreds of thousands of years.",
    "dog": "A dog is a domesticated mammal and one of the most popular pets worldwide. Dogs are known for their loyalty and are used as companions, working animals, and service animals.",
    "cat": "A cat is a small domesticated carnivorous mammal commonly kept as a pet. Cats are known for their agility, independence, and hunting instincts.",
    "human": "A human is a member of the species Homo sapiens, the most advanced primate on Earth. Humans are distinguished by their intelligence, use of language, and ability to create complex societies.",
    "man": "A man is an adult male human being. Men, like all humans, are members of the species Homo sapiens and play various roles in society, culture, and family life.",
    "woman": "A woman is an adult female human being. Women, like all humans, are members of the species Homo sapiens and contribute to all areas of society including science, politics, and arts.",
    "child": "A child is a young human being below the age of puberty or below the legal age of majority. Children learn and develop rapidly through play, education, and social interaction.",
    "school": "A school is an institution where students receive education under the guidance of teachers. Schools provide structured learning environments for children and young adults.",
    "money": "Money is a medium of exchange used to buy goods and services. It exists in the form of coins, banknotes, and digital currency and is essential to modern economies.",
    "time": "Time is the progression of events from the past through the present into the future. It is measured in seconds, minutes, hours, days, and years.",
    "love": "Love is a strong feeling of deep affection and care for another person or thing. It is one of the most fundamental human emotions and exists in many forms including romantic, familial, and platonic love.",
    "music": "Music is an art form consisting of organized sound and rhythm. It is one of the oldest human art forms and plays an important role in culture, emotion, and communication.",
    "sport": "Sport is a physical activity involving skill and competition, typically governed by rules. Sports promote fitness, teamwork, and discipline and are enjoyed by billions worldwide.",
    "game": "A game is a structured form of play with rules and objectives, enjoyed for entertainment or competition. Games can be physical, mental, or digital and are played by people of all ages.",
    "art": "Art is the expression of human creativity and imagination through visual, performing, or literary works. It includes painting, sculpture, music, literature, and many other forms.",
    "science": "Science is the systematic study of the natural world through observation and experimentation. It produces knowledge in fields like physics, chemistry, biology, and astronomy.",
    "health": "Health is the state of complete physical, mental, and social wellbeing, not merely the absence of disease. Good health is maintained through proper nutrition, exercise, and medical care.",
    "education": "Education is the process of learning and acquiring knowledge, skills, values, and attitudes. It takes place through formal institutions like schools and universities and through life experience.",
    "internet": "The Internet is a global network of computers and devices connected together, enabling communication and information sharing worldwide. It has transformed how people work, communicate, and access information.",
    "electricity": "Electricity is a form of energy resulting from the flow of electric charge. It powers homes, industries, and devices and is generated from sources like coal, solar, wind, and nuclear energy.",
    "language": "Language is a system of communication using words, symbols, or sounds that humans use to express thoughts and feelings. There are over 7,000 languages spoken around the world.",
    "religion": "Religion is a system of beliefs, practices, and moral values that relates humanity to spiritual or supernatural elements. Major world religions include Christianity, Islam, Hinduism, and Buddhism.",
    "government": "A government is the system or group of people that governs a state or community. Governments make laws, maintain order, provide services, and represent their citizens.",
    "economy": "An economy is the system of production, distribution, and consumption of goods and services in a society. It encompasses trade, industry, finance, and labor.",
    "culture": "Culture is the shared beliefs, values, customs, arts, and way of life of a group of people. It is passed down through generations and shapes how communities think and behave.",
    "history": "History is the study of past events, particularly human affairs. It helps us understand how societies developed and how past decisions have shaped the present world.",
    "mathematics": "Mathematics is the study of numbers, quantities, shapes, and patterns. It is used in science, engineering, finance, and everyday life for problem solving and logical reasoning.",
    "medicine": "Medicine is the science and practice of diagnosing, treating, and preventing disease. It encompasses various specialties and has extended human life expectancy significantly.",
    "energy": "Energy is the capacity to do work or produce change. It exists in many forms including kinetic, potential, thermal, and electrical energy and is fundamental to all physical processes.",
    "environment": "The environment refers to the natural world including air, water, land, plants, and animals. Protecting the environment is important for the health of all living things on Earth.",
    "technology": "Technology is the application of scientific knowledge for practical purposes. It includes tools, machines, software, and systems that improve how humans live and work.",
    "democracy": "Democracy is a system of government where power is held by the people, either directly or through elected representatives. It values freedom, equality, and the rule of law.",
    "africa": "Africa is the world's second largest and second most populous continent. It is home to 54 recognized countries and over 1.4 billion people speaking thousands of languages.",
    "nigeria": "Nigeria is a country in West Africa and the most populous country on the African continent with over 220 million people. Its capital is Abuja and its largest city is Lagos.",
    "america": "America refers to the United States of America, a country in North America with 50 states. It is the world's largest economy and a major global political and cultural influence.",
    "europe": "Europe is a continent located in the Northern Hemisphere, bordered by the Atlantic Ocean to the west and Asia to the east. It is home to 44 countries and about 750 million people.",
    "asia": "Asia is the world's largest continent by both area and population. It is home to over 4.7 billion people and includes countries like China, India, Japan, and many others.",
    "gravity": "Gravity is the force that attracts objects with mass toward one another. On Earth it pulls objects toward the center of the planet giving them weight. It keeps planets in orbit around the Sun.",
    "atom": "An atom is the smallest unit of a chemical element. It consists of a nucleus containing protons and neutrons, surrounded by electrons. Everything in the universe is made of atoms.",
    "oxygen": "Oxygen is a chemical element essential for life. It makes up about 21 percent of Earth's atmosphere and is required for breathing and combustion.",
    "blood": "Blood is the red fluid that circulates through the body, carrying oxygen and nutrients to cells and removing waste products. It is composed of red blood cells, white blood cells, platelets, and plasma.",
    "brain": "The brain is the organ that controls thought, memory, emotion, and body functions. It is the most complex organ in the human body and is protected by the skull.",
    "heart": "The heart is a muscular organ that pumps blood throughout the body. It beats about 100,000 times per day and is essential for life.",
    "planet": "A planet is a large celestial body that orbits a star. Our solar system has eight planets including Earth, Mars, Jupiter, and Saturn.",
    "star": "A star is a luminous ball of gas held together by gravity. Stars produce light and heat through nuclear fusion. The Sun is the closest star to Earth.",
    "ocean": "An ocean is a vast body of saltwater covering most of the Earth's surface. There are five oceans on Earth including the Pacific, Atlantic, Indian, Arctic, and Southern oceans.",
    "mountain": "A mountain is a large natural elevation of rock and earth rising above the surrounding land. Mountains are formed by tectonic forces and erosion over millions of years.",
    "river": "A river is a large natural stream of water flowing toward an ocean, lake, or other river. Rivers are important for freshwater supply, transportation, and ecosystems.",
    "forest": "A forest is a large area of land covered with trees and undergrowth. Forests are home to the majority of Earth's biodiversity and play a crucial role in regulating climate.",
    "desert": "A desert is a dry barren area of land with very little rainfall. Deserts cover about one third of Earth's land surface and can be hot or cold.",
    "virus": "A virus is a tiny infectious agent that can only reproduce inside living cells. Viruses cause diseases like the flu, common cold, and COVID-19.",
    "bacteria": "Bacteria are microscopic single-celled organisms found nearly everywhere on Earth. Some cause disease while others are beneficial to humans and the environment.",
    "evolution": "Evolution is the process by which species change over generations through natural selection. It explains the diversity of life on Earth and was described by Charles Darwin.",
    "photosynthesis": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce food and oxygen. It is fundamental to life on Earth.",
    "earthquake": "An earthquake is the shaking of the Earth's surface caused by the movement of tectonic plates. Earthquakes can cause significant damage and may trigger tsunamis.",
    "volcano": "A volcano is an opening in the Earth's crust through which lava, ash, and gases escape. Volcanic eruptions can be highly destructive but also create new land.",
    "democracy": "Democracy is a system of government where citizens exercise power through elected representatives. It is based on principles of freedom, equality, and majority rule.",
    "constitution": "A constitution is a set of fundamental laws and principles that establish the framework of a government. It defines the rights of citizens and the powers of the state.",
    "hospital": "A hospital is a health care institution providing patient treatment by specialized staff and equipment. Hospitals treat injuries and illnesses and provide emergency care.",
    "bank": "A bank is a financial institution that accepts deposits, provides loans, and offers other financial services. Banks are essential to modern economies.",
    "market": "A market is a place where buyers and sellers come together to trade goods and services. Markets can be physical locations or online platforms.",
    "election": "An election is a formal process in which people vote to choose their leaders or decide on policies. Elections are the foundation of democratic governance.",
    "war": "War is an armed conflict between nations, states, or groups. Wars have shaped human history and result in significant loss of life and destruction.",
    "peace": "Peace is a state of harmony and absence of conflict or violence. It is one of the most fundamental human aspirations and is essential for development.",
    "freedom": "Freedom is the power to act, speak, or think without restraint. It is a fundamental human right recognized in most modern constitutions and international law.",
    "justice": "Justice is the quality of being fair and reasonable in the treatment of people. It is a cornerstone of legal systems and moral philosophy worldwide.",
    "poverty": "Poverty is the state of lacking sufficient resources to meet basic needs such as food, shelter, and clothing. It affects billions of people worldwide.",
    "climate": "Climate is the long-term pattern of weather in a particular area. Climate change refers to shifts in global temperatures and weather patterns caused largely by human activity.",
    "football": "Football is one of the world's most popular sports, played between two teams of eleven players using a round ball. The objective is to score goals by getting the ball into the opposing team's net.",
    "basketball": "Basketball is a sport played between two teams of five players who score points by throwing a ball through a hoop. It was invented by James Naismith in 1891.",
    "laptop": "A laptop is a portable personal computer that is small enough to use on your lap. It combines a screen, keyboard, and computer components in one device.",
    "television": "A television is an electronic device that receives and displays moving images and sound. It is one of the most common forms of entertainment and news distribution worldwide.",
    "radio": "A radio is a device that receives electromagnetic waves and converts them into sound. Radio broadcasting has been a major form of communication and entertainment since the early 20th century.",
    "airplane": "An airplane is a powered flying vehicle with fixed wings. It is used for transporting passengers and cargo across long distances and has transformed global travel.",
    "ship": "A ship is a large watercraft designed for ocean or sea travel. Ships are used for transporting goods and passengers across the world's oceans.",
    "train": "A train is a form of rail transport consisting of a series of connected vehicles running on tracks. Trains are used for both passenger travel and cargo transportation.",
    "bicycle": "A bicycle is a human-powered vehicle with two wheels. It is one of the most efficient forms of transportation and is used for commuting, recreation, and sport.",
    "hospital": "A hospital is a healthcare institution providing medical treatment and care to patients. It is staffed by doctors, nurses, and other healthcare professionals.",
    "university": "A university is a higher education institution that awards degrees and conducts research. Universities offer undergraduate and postgraduate programs across many fields.",
    "library": "A library is a place where books, magazines, and other resources are kept for people to read or borrow. Libraries are important centers of learning and community.",
    "museum": "A museum is an institution that collects, preserves, and displays objects of historical, scientific, or artistic significance. Museums educate the public about culture and history.",
    "prison": "A prison is a facility where people convicted of crimes are held as punishment. Prisons are part of the criminal justice system and aim to rehabilitate offenders.",
    "police": "The police are a government organization responsible for maintaining law and order, preventing crime, and protecting citizens. Police officers enforce laws and respond to emergencies.",
    "army": "An army is a large organized military force trained for land warfare. Armies are used to defend nations and may also assist in disaster relief and peacekeeping operations.",
    "medicine": "Medicine is the science and practice of preventing, diagnosing, and treating disease. It encompasses a wide range of specialties from surgery to psychiatry.",
    "surgery": "Surgery is a medical procedure involving manual or operative techniques to treat injuries, diseases, or deformities. It is performed by trained surgeons in hospitals.",
    "vaccine": "A vaccine is a biological preparation that provides immunity to a specific disease. Vaccines have eradicated or greatly reduced many deadly diseases worldwide.",
    "antibiotic": "An antibiotic is a type of medicine that kills or inhibits the growth of bacteria. Antibiotics have saved millions of lives since their discovery in the 20th century.",
    "nutrition": "Nutrition is the process of obtaining and using food for growth, energy, and health. Good nutrition is essential for physical and mental wellbeing.",
    "exercise": "Exercise is physical activity performed to improve health and fitness. Regular exercise reduces the risk of disease, improves mood, and increases lifespan.",
    "sleep": "Sleep is a natural state of rest in which the body and mind recover. Adults need 7 to 9 hours of sleep per night for optimal health and cognitive function.",
    "stress": "Stress is the body's response to demanding or threatening situations. Chronic stress can lead to health problems including anxiety, depression, and heart disease.",
    "depression": "Depression is a mental health condition characterized by persistent sadness, loss of interest, and lack of energy. It is one of the most common mental disorders worldwide.",
    "anxiety": "Anxiety is a feeling of worry, nervousness, or unease about something with an uncertain outcome. Anxiety disorders are the most common mental health conditions globally.",
}

def lookup_builtin(question: str) -> dict | None:
    """Check built-in dictionary for common everyday words."""
    q = question.lower().strip().rstrip("?").strip()
    for prefix in [
        "what is a ", "what is an ", "what is the ", "what is ",
        "what are ", "define ", "explain ", "tell me about ",
        "describe ", "who is a ", "who is an ",
    ]:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break
    # Direct match
    if q in BUILTIN_ANSWERS:
        return {
            "answer": BUILTIN_ANSWERS[q],
            "confidence": 0.85,
            "matched_question": question,
            "category": "general",
            "source": "knowledge_base",
        }
    # Partial match
    for key, answer in BUILTIN_ANSWERS.items():
        if key in q or q in key:
            return {
                "answer": answer,
                "confidence": 0.75,
                "matched_question": question,
                "category": "general",
                "source": "knowledge_base",
            }
    return None
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

def compute_tf(tokens):
    count = Counter(tokens)
    total = len(tokens) or 1
    return {term: freq / total for term, freq in count.items()}

def compute_idf(corpus):
    N = len(corpus)
    idf = {}
    all_terms = set(t for doc in corpus for t in doc)
    for term in all_terms:
        df = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log((N + 1) / (df + 1)) + 1
    return idf

def tfidf_vector(tokens, idf):
    tf = compute_tf(tokens)
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}

def cosine_similarity(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

_corpus_tokens = [preprocess(item["question"]) for item in knowledge_base]
_idf = compute_idf(_corpus_tokens)
_doc_vectors = [tfidf_vector(tokens, _idf) for tokens in _corpus_tokens]

def find_best_answer(question, top_k=1):
    q_tokens = preprocess(question)
    q_vec = tfidf_vector(q_tokens, _idf)
    scored = sorted(
        [(cosine_similarity(q_vec, doc_vec), idx)
         for idx, doc_vec in enumerate(_doc_vectors)],
        reverse=True
    )
    return [{
        "question": knowledge_base[idx]["question"],
        "answer": knowledge_base[idx]["answer"],
        "category": knowledge_base[idx].get("category", "general"),
        "confidence": round(score, 4),
    } for score, idx in scored[:top_k]]


def extract_topic(question: str) -> str:
    q = question.lower().strip()
    patterns = [
        r"^what is (a |an |the )?",
        r"^what are (a |an |the )?",
        r"^who is (a |an |the )?",
        r"^who was (a |an |the )?",
        r"^how does (a |an |the )?",
        r"^how do (a |an |the )?",
        r"^where is (a |an |the )?",
        r"^when (was|is|did) (a |an |the )?",
        r"^define (a |an |the )?",
        r"^tell me about (a |an |the )?",
        r"^explain (a |an |the )?",
        r"^describe (a |an |the )?",
    ]
    for pattern in patterns:
        q = re.sub(pattern, "", q)
    return q.strip().rstrip("?").strip()


def clean_answer(text: str, max_sentences: int = 3) -> str:
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    result = '. '.join(sentences[:max_sentences])
    if result and not result.endswith('.'):
        result += '.'
    return result


async def search_wikipedia(topic: str, client: httpx.AsyncClient) -> str | None:
    try:
        search_res = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": topic,
                "limit": "5",
                "format": "json",
            },
            timeout=8.0,
        )
        data = search_res.json()
        titles = data[1] if len(data) > 1 else []
        if not titles:
            return None
        for title in titles:
            summary_res = await client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                timeout=8.0,
                follow_redirects=True,
            )
            if summary_res.status_code != 200:
                continue
            page = summary_res.json()
            if page.get("type") == "disambiguation":
                continue
            extract = page.get("extract", "")
            if extract and len(extract) > 40:
                return clean_answer(extract)
    except Exception as e:
        print(f"Wikipedia opensearch error: {e}")
    return None


async def ask_wikipedia(question: str) -> dict | None:
    try:
        async with httpx.AsyncClient() as client:
            topic = extract_topic(question)
            answer = await search_wikipedia(topic, client)
            if not answer:
                answer = await search_wikipedia(question, client)
            if answer:
                return {
                    "answer": answer,
                    "confidence": 0.68,
                    "matched_question": question,
                    "category": "general",
                    "source": "wikipedia",
                }
    except Exception as e:
        print(f"Wikipedia error: {e}")
    return None


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 1

    @validator("question")
    def question_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or less")
        return v


class AskResponse(BaseModel):
    answer: str
    confidence: float
    matched_question: str
    category: str
    source: str = "knowledge_base"


@app.get("/")
def root():
    return {"status": "ok", "kb_size": len(knowledge_base)}

@app.get("/health")
def health():
    return {"status": "healthy", "kb_entries": len(knowledge_base)}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    # Step 1 — Knowledge base (instant, always free)
    results = find_best_answer(payload.question)
    if results and results[0]["confidence"] >= 0.15:
        best = results[0]
        return AskResponse(
            answer=best["answer"],
            confidence=best["confidence"],
            matched_question=best["question"],
            category=best["category"],
            source="knowledge_base",
        )

    # Step 2 — Built-in dictionary (no internet needed)
    builtin_result = lookup_builtin(payload.question)
    if builtin_result:
        return AskResponse(**builtin_result)

    # Step 3 — Wikipedia (free, handles almost anything)
    wiki_result = await ask_wikipedia(payload.question)
    if wiki_result:
        return AskResponse(**wiki_result)

    # Step 4 — Nothing found
    raise HTTPException(
        status_code=404,
        detail="No answer found. Try rephrasing your question.",
    )


@app.get("/questions")
def list_questions(category: Optional[str] = None):
    data = knowledge_base
    if category:
        data = [item for item in data if item.get("category") == category]
    return {"count": len(data), "questions": [item["question"] for item in data]}