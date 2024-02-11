import streamlit as st

st.title('🎈 Banky McBankFace')

# User input
user_input = st.text_input("Type your message here...", key="user_input")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I assist you today?"}]

# Button to send message
if st.button('Send'):
    if user_input:  # Check if input is not empty
        chat_response = collect_and_process_messages(user_input)
        # Update UI with the response
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": chat_response})
        # Clear input box (This part is handled automatically in Streamlit)
    else:
        st.error("Please enter a message.")

