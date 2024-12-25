# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Define FAQs as a dictionary
faqs = {
    "What is the product's price?": "The product's price is $99.99.",
    "What are the product's features?": "The product has features X, Y, and Z.",
    "How do I return the product?": "Please contact our customer service team.",
}

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stopwords."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in stopwords.words("english")]
    return " ".join(tokens)

def find_match(text, faqs):
    """Find the best match for the input text in the FAQs."""
    best_match = None
    best_score = 0
    for question, answer in faqs.items():
        # Use spaCy to calculate similarity between input text and FAQ question
        similarity = nlp(text).similarity(nlp(question))
        if similarity > best_score:
            best_match = answer
            best_score = similarity
    return best_match

def generate_response(text):
    """Generate a response based on the input text."""
    text = preprocess_text(text)
    answer = find_match(text, faqs)
    if answer:
        return answer
    else:
        return "Sorry, I didn't understand your question."

# Test the chatbot
while True:
    user_input = input("User: ")
    response = generate_response(user_input)
    print("Chatbot:", response)
