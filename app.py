import streamlit as st
import re
from generator import StrataGenerator

st.set_page_config(page_title="Strata AI", page_icon="🏛️", layout="wide")

@st.cache_resource(show_spinner="Waking up the RTX 5060... (This only happens once per server run)")
def load_pipeline():
    return StrataGenerator(model_path="strata-qwen-7b-archaeology-lora-production")

generator = load_pipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"Session 1": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"
if "session_counter" not in st.session_state:
    st.session_state.session_counter = 1

with st.sidebar:
    st.title("🏛️ Strata Archives")
    
    if st.button("➕ New Excavation (New Chat)", use_container_width=True):
        st.session_state.session_counter += 1
        new_session_name = f"Session {st.session_state.session_counter}"
        st.session_state.chat_history[new_session_name] = []
        st.session_state.current_session = new_session_name
        st.rerun()
        
    st.divider()
    st.subheader("Past Chats")
    
    for session_name in reversed(list(st.session_state.chat_history.keys())):
        if st.button(session_name, use_container_width=True):
            st.session_state.current_session = session_name
            st.rerun()

st.title(f"🏛️ Strata RAG - {st.session_state.current_session}")

current_messages = st.session_state.chat_history[st.session_state.current_session]

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about global archaeology..."):
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing text and running inference..."):
            
            # Clean the prompt of punctuation to match intents perfectly
            clean_prompt = re.sub(r'[^\w\s]', '', prompt.lower().strip())
            casual_triggers = {"hi", "hello", "hey", "greetings", "who are you", "what are you", "help", "how can you help me", "what can you do"}
            
            if clean_prompt in casual_triggers:
                override_context = "Ignore context. Introduce yourself politely as Strata, an expert archaeological AI assistant. Keep it brief."
                answer = generator.generate_answer(prompt, override_context)
                
                st.markdown(answer)
                current_messages.append({"role": "assistant", "content": answer})
                
            else:
                context_docs = generator.retriever.search(prompt, top_k=3)
                
                context_strings = []
                for doc in context_docs:
                    if isinstance(doc, dict):
                        context_strings.append(doc.get("text", doc.get("content", str(doc))))
                    else:
                        context_strings.append(doc)
                        
                context = "\n".join(context_strings)
                answer = generator.generate_answer(prompt, context)
                
                st.markdown(answer)
                current_messages.append({"role": "assistant", "content": answer})
