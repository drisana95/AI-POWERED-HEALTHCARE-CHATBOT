from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# Load the pre-trained model and tokenizer
model_name = "gpt2"  # You can also use "EleutherAI/gpt-neo-2.7B" for a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    # Create a well-structured prompt
    structured_prompt = f"Explain in detail: {prompt}"
    
    inputs = tokenizer(structured_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=150,
        num_return_sequences=1,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def main():
    st.title("Healthcare Chatbot")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    user_input = st.text_input("You:", "")

    if st.button("Send"):
        if user_input:
            # Add user input to chat history
            st.session_state['history'].append({"role": "user", "content": user_input})
            
            # Generate a response using the model
            response = generate_response(user_input)
            
            # Add the response to chat history
            st.session_state['history'].append({"role": "assistant", "content": response})

    # Display chat history
    for chat in st.session_state['history']:
        if chat['role'] == 'user':
            st.write(f"You: {chat['content']}")
        else:
            st.write(f"Chatbot: {chat['content']}")

# Correct the __name__ check
if __name__ == "__main__":
    main()
