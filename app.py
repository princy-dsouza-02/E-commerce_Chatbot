import streamlit as st

from chatbot import EcommerceChatbot

# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="E-Commerce AI Chatbot",
    page_icon="🛒",
    layout="centered" 
)

# ------------------------------
# Custom CSS 
# ------------------------------
st.markdown("""
<style>

    /* --------------------------------
       Heading above message box
       -------------------------------- */

    .help-text {
        font-size: 22px;
        font-weight: 600;
        color: black;
        margin-bottom: 8px;
    }

    /* --------------------------------
       Chat input box
       -------------------------------- */

    div[data-testid="stChatInput"] {
        border: 2px solid black !important;
        border-radius: 10px !important;
        background-color: white !important;
    }

    /* Text inside input */

    div[data-testid="stChatInput"] textarea {
        color: black !important;
        background-color: white !important;
        border: none !important;
        outline: none !important;
    }

    /* --------------------------------
       Send button
       -------------------------------- */

    div[data-testid="stChatInput"] button {
        border: 2px solid black !important;
        border-radius: 50% !important;

        background-color: white !important;
        color: black !important;

        width: 36px !important;
        height: 36px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        margin-right: 5px !important;
    }

    /* Hide Streamlit's original send icon */

    div[data-testid="stChatInput"] button svg {
        display: none !important;
    }

    /* Add arrow */

    div[data-testid="stChatInput"] button::after {
        content: "➤";
        font-size: 20px;
        color: black;
        line-height: 1;
    }

    /* Arrow button hover */

    div[data-testid="stChatInput"] button:hover {
        background-color: #eeeeee !important;
        border: 2px solid black !important;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Title
# ---------------------------------
st.title("🛒 E-Commerce Customer Support Chatbot")

st.write(
    "Ask questions about orders, delivery, payments, "
    "returns and refunds."
)

# --------------------------------
# Load chatbot
# --------------------------------
@st.cache_resource
def load_chatbot():
    return EcommerceChatbot()

bot = load_chatbot()

# -------------------------------
# Chat history
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ----------------------------------
# User input
# ----------------------------------
st.markdown(
    '<div class="help-text">How can I help you today?</div>',
    unsafe_allow_html=True
)

query = st.chat_input(
    "Type your message here..."
)

if query:

    # Display user message
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Get chatbot result
    result = bot.answer(query)

    # Display chatbot response
    with st.chat_message("assistant"):

        st.write(result["response"])

        st.divider()

        st.write(
            "**Detected Category:**",
            result["category"]
        )

        st.write(
            "**Detected Intent:**",
            result["intent"]
        )

        if result["entities"]:

            st.write("**Important Information:**")

            for key, values in result["entities"].items():

                if isinstance(values, list):
                    values = ", ".join(str(v) for v in values)

                display_names = {
                    "order_id": "Order ID",
                    "tracking_number": "Tracking Number",
                    "email": "Email",
                    "phone": "Phone Number"
                }

                display_key = display_names.get(
                    key,
                    key.replace("_", " ").title()
                )

                st.write(
                    f"**{display_key}:** {values}"
                )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"]
    })
    