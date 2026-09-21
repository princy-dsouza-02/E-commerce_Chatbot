import re
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from responses import RESPONSES

class EcommerceChatbot:

    def __init__(self):

        # Load embedding model
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load FAISS index
        self.index = faiss.read_index(
            "models/retail_faiss.index"
        )

        # Load dataset
        with open("models/retail_data.pkl", "rb") as f:
            self.df = pickle.load(f)

    # --------------------------------------------------
    # 1. ENTITY EXTRACTION
    # --------------------------------------------------
    def extract_entities(self, text):

        entities = {}
   
        # Order ID / Order Number
        order_pattern = (
            r'(?:order\s*(?:id|number)|order\s*#)'
            r'\s*(?:is|:|#|-)?\s*([A-Za-z0-9-]+)'
        )

        order_numbers = re.findall(
            order_pattern,
            text,
            re.IGNORECASE
        )

        if order_numbers:
            entities["order_id"] = order_numbers

        # Tracking Number
        tracking_pattern = (
            r'(?:tracking\s*(?:number|id)?|tracking)'
            r'\s*(?:is|:|#|-)?\s*([A-Za-z0-9-]+)'
        )

        tracking_numbers = re.findall(
            tracking_pattern,
            text,
            re.IGNORECASE
        )

        if tracking_numbers:
            entities["tracking_number"] = tracking_numbers

        # Email
        emails = re.findall(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            text
        )

        if emails:
            entities["email"] = emails

        # Phone number
        phones = re.findall(
            r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b',
            text
        )

        if phones:
            entities["phone"] = phones

        return entities
    
    # --------------------------------------------------
    # 2.SEMANTIC SEARCH
    # --------------------------------------------------
    def semantic_search(self, query, top_k=3):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        # Normalize
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            row = self.df.iloc[idx]

            results.append({
                "score": float(score),
                "instruction": row["instruction"],
                "category": row["category"],
                "intent": row["intent"],
                "response": row["response"]
            })

        return results

    # ------------------------------------------------
    # 3.CATEGORY MAPPING
    # ------------------------------------------------
    def get_category(self, intent):

        category_map = {

            "track_order": "ORDER",

            "track_delivery": "ORDER",

            "cancel_order": "ORDER",

             "return_product": "RETURNS",

            "refund_status": "RETURNS",

            "refund_policy": "RETURNS",

            "payment_issue": "PAYMENT"
        }

        return category_map.get(
            intent,
            "OTHER"
        )

    # ------------------------------------------------
    # 4. FOLLOW-UP RESPONSE
    # ------------------------------------------------
    def get_followup_response(self, intent, entities):

        # Refund status
        if intent == "refund_status":

            if "order_id" in entities:

                order_id = entities["order_id"][0]

                return (
                    f"Thank you. I have received your order ID "
                    f"{order_id}. Your refund request can now be "
                    f"checked using this order information."
                )

            return (
                "I can help you check your refund status. "
                "Please provide your order ID."
            )

        # Track order
        if intent == "track_order":

            if "order_id" in entities:

                order_id = entities["order_id"][0]

                return (
                    f"Thank you. I have received order ID "
                    f"{order_id}. It can be used to check your "
                    f"order status."
                )

            if "tracking_number" in entities:

                tracking = entities["tracking_number"][0]

                return (
                    f"Thank you. I have received tracking number "
                    f"{tracking}. It can be used to check your "
                    f"delivery status."
                )

        # Return product
        if intent == "return_product":

            if "order_id" in entities:

                order_id = entities["order_id"][0]

                return (
                    f"Thank you. I have received order ID "
                    f"{order_id}. I can use it to continue "
                    f"your return request."
                )

        # Cancel order
        if intent == "cancel_order":

            if "order_id" in entities:

                order_id = entities["order_id"][0]

                return (
                    f"Thank you. I have received order ID "
                    f"{order_id}. I can use it to continue "
                    f"your cancellation request."
                )


        return (
            "Thank you. I have received the information."
        )

    # =================================================
    # 5. MAIN ANSWER FUNCTION
    # =================================================
    def answer(self, query, previous_intent=None):
     
        # Extract entities
        entities = self.extract_entities(query)

        # Follow -up Message
        if previous_intent and entities:
        
            followup_intents = [
                "refund_status",
                "return_product",
                "cancel_order",
                "track_order",
                "track_delivery"
            ]
        
            if previous_intent in followup_intents:
        
                return {
                    "query": query,
                    "category": self.get_category(previous_intent),
                    "intent": previous_intent,
                    "confidence": 1.0,
                    "entities": entities,
                    "response": self.get_followup_response(
                        previous_intent,
                        entities
                        ),
                    "results": []

                }

        
        # Normal Semantic Search
        results = self.semantic_search(
            query,
            top_k=3
        )
        
        best = results[0]
        
        # Low Confidence 
        if best["score"] < 0.40:
            return {
                "query": query,
                "category": "UNKNOWN",
                "intent": "unknown",
                "confidence": best["score"],
                "entities": entities,
                "response": (
                    "I'm not sure I understood your request. "
                    "Could you please provide more details?"
                ),
                "results": results
            }

        # Create Response 
        # If refund status and order ID already provided
        if best["intent"] == "refund_status":

            if "order_id" in entities:

                order_id = entities["order_id"][0]

                response = (
                    f"Thank you. I have received your "
                    f"order ID {order_id}. Your refund "
                    f"request can now be checked using "
                    f"this order information."
                )

            else:

                response = (
                    "I can help you check your refund "
                    "status. Please provide your order ID."
                )

        # If tracking order and order ID already provided
        elif best["intent"] == "track_order":

            if "order_id" in entities:

                order_id = entities["order_id"][0]

                response = (
                    f"Thank you. I have received your "
                    f"order ID {order_id}. It can be used "
                    f"to check your order status."
                )

            elif "tracking_number" in entities:

                tracking = entities["tracking_number"][0]

                response = (
                    f"Thank you. I have received tracking "
                    f"number {tracking}. It can be used "
                    f"to check your delivery status."
                )

            else:

                response = (
                    "I can help you track your order. "
                    "Please provide your order ID or "
                    "tracking number."
                )

        # Other intents
        else:

            response = best["response"]

        # Return Result
        return {
                "query": query,
                "category": best["category"],
                "intent": best["intent"],
                "confidence": best["score"],
                "entities": entities,
                "response": response,
                "results": results
        }
            
# ------------------------------------------------------
# Test chatbot
# ------------------------------------------------------
if __name__ == "__main__":

    bot = EcommerceChatbot()

    while True:

        query = input("\nCustomer: ")

        if query.lower() == "exit":
            break

        result = bot.answer(query)

        print("\nCategory:", result["category"])
        print("Intent:", result["intent"])
        print("Similarity:", result["confidence"])
        print("Entities:", result["entities"])
        print("\nBot:", result["response"])
