from sentence_transformers import SentenceTransformer, util

intents = [
    {
        "name": "Fallback",
        "questions": [],
        "answer":
            (lambda: "I don't understand. May you repeat with different words?"),
    },
    {
        "name": "Welcome",
        'questions': [
            "Привет",
            "Как дела",
            "Доброе утро",
            "Здравствуй",
        ],
        "answer": (lambda: "Hi! How can I help you?"),
    },
];


class PredictionService(object):
    def __init__(self):
        self.embedder = SentenceTransformer('inkoziev/sbert_synonymy')
        self.corpus = []

        self.prepare_corpus()

    def prepare_corpus(self):
        from_corpus_id_to_intent_id = {}
        corpus_id = 0
        for intent_id, intent in enumerate(intents):
            for question in intent["questions"]:
                self.corpus.append(question)
                from_corpus_id_to_intent_id[corpus_id] = intent_id
                corpus_id += 1

        return from_corpus_id_to_intent_id

    def get_embed_corpus(self):
        return self.embedder.encode(self.corpus, convert_to_tensor=True)

    def get_matched_intent(self, query, corpus_embeddings, from_corpus_id_to_intent_id, min_score_threshold=0.6):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hit = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)[0][0]
        score = hit['score']
        closest_question = self.corpus[hit['corpus_id']]
        intent_id = from_corpus_id_to_intent_id[hit['corpus_id']]
        matched_intent = intents[intent_id]

        if score >= min_score_threshold:
            return matched_intent, score, closest_question

        return intents[0], 0, ""
