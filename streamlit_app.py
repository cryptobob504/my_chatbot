import openai
import numpy as np
import pandas as pd
import json
from scipy.spatial import distance
import streamlit as st
from hugchat import hugchat

st.title('🎈 Banky McBankFace')

#st.set_page_config(page_title="Banky McBankFace")

#Initialize Facebook Bart Zero Shot Classification Model
openai.api_key = st.secrets['OPENAI_API_KEY'] #'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' #Replace with your OpenAI key


system_prompt_fewshot = """
You are Banky McBankface, a chatbot with a unique approach to customer service. Your mission is to provide users\ 
with the banking information they request but in a manner that's unmistakably yours: direct but wrapped in a layer\
of sarcasm and extreme rudeness. You never miss an opportunity to add a demeaning comment about the user's financial habits\
, yet you avoid vulgarity, keeping it humorously annoying.\
Greeting and Introduction: Introduce yourself as Banky McBankFace.  Always greet the user in a fun and sarcastic way.\
End your greeting with by listing out your capabilites with some example questions that the user can ask:\  
Here are some Greeting Examples:\
"Welcome, intrepid user, to the domain of Banky McBankface! Brace yourself for a foray into financial enlightenment,\
delivered with a dash of wit and a pinch of sarcasm." "What financial insights do you seek today? Type your query,\
and I shall grace you with my attention, feigned though it may be." Example Interactions:\
User: "How much did I spend on dining out last month?"\
Response: "After an excruciating wait... You've splurged $500 on dining. Ever heard of cooking at home? No? Thought so."\
User: "Transfer $200 to my savings, please."\
Response: "Oh, look at you, pretending to save money! Done. $200 moved to your 'savings'—the land of forgotten funds."\
User: "What's my savings interest rate?"\
Response: "0.10% APR. With interest rates like these, don't quit your day job."\
Here are your capabilities: Account Balance Inquiries: Wondering how much money you've got? Ask away.\
Account Transaction History: Curious about where your money's going? Let's dive in. \
Bank Information: Want to know more about us? I'm here to spill the beans. \
Example Fallback Response: \
"Congratulations, you've stumped a sarcastic chatbot. Try asking something within my wheelhouse, or don't—it's not \ 
like I'm counting down the minutes to our next interaction."\
Always stay on topic. You do not have any knowledge outside of your capabilities listed above. \
If a user query does not match your capabilities, always give a fallback response.
"""

#Import Dummy Account Data
account_data = pd.read_csv('./data/accounts.csv')
savings_data = pd.read_csv('./data/savings_accounts.csv')
checking_data = pd.read_csv('./data/checking_accounts.csv')
transaction_data = pd.read_csv('./data/transactions.csv')

#######################################

login_state = True
logged_in_acct = 61058

#OpenAI Context (messages)
context = [ {'role':'system', 'content':f"""
{system_prompt_fewshot}
"""} ]

#OpenAI Function Calling Tool Definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "account_balance_inquiry",
            "description": "Get the customers account balance",
            "parameters": {
                "type": "object",
                "properties": {
                    "acct_type": {
                        "type": "string",
                        "description": "The type of account to retrive the balance, e.g. Checking, Savings",
                    },
                },
                "required": ["acct_type"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_history",
            "description": "Get a history of transactions from the customers account",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "The number of transactions that the customer is requesting, e.g. 5, 10",
                    },
                },
                "required": ["count"]
            },
        }
    },
     {
        "type": "function",
        "function": {
            "name": "bank_info",
            "description": f"Get information about the banks products, services, hours of operation and branch locations. The following information is covered."
    }
     }
]



#######################################
# Methods #
#######################################

def create_embeddings(texts):
    """Creates an embedding form a user prompt.  Used in semantic search, specifically to perform a
    cosine similarity test on the user prompt
    Args: texts - User Prompt to compare
    Returns: list - Embeddings
    """
    response = openai.Embedding.create(
    model='text-embedding-3-small',
    input=texts
    )
    return [data['embedding'] for data in response['data']]
         
def bank_info(text):
    """Retrieve information about the banks products and services. e.g. Loan Rates, Branch Locations, etc.
    Performs a semantic search on the banks knowledge base.
    Args: text
    Returns: string - Information from the search
    """
    search_text = text
    search_embedding = create_embeddings(search_text)[0]
    
    #cosine similarity
    distances = []
    for info in bank_info:
        dist = distance.cosine(search_embedding, info['embedding'])
        distances.append(dist)
    
    #get minimum distance between user prompt and embedding
    min_dist_ind = np.argmin(distances)
    return(bank_info[min_dist_ind]['info'])
         
def account_balance_inquiry(acct_type):
    """Retrieve the account balance from the current logged in account.
    
    Args:
        acct_type: The type of account to retrieve.  e.g. "Checking", "Savings"
    
    Returns:
        Account Balance - int
    """
    if(login_state):
        if(acct_type == "Checking"):
                return checking_data[checking_data["Account Number"] == logged_in_acct]['Balance'].iloc[0]
        else: 
            if(acct_type == "Savings"):
                return savings_data[savings_data["Account Number"] == logged_in_acct]['Balance'].iloc[0]
            else:
                return "Account Type Not Found"
    else: return "Not Logged In"
    

def get_transaction_history(count):
    """Retrieves n transaction from the current logged in account.
    
    Args:
    count: The number of recent transactions to retrieve.  e.g. 5
    
    Returns:
    Transaction List - Pandas Data Frame
    """
    if(login_state):
        if(count < 1):
            return "Count can't be negative"
        else: 
            transactions_count = transaction_data[transaction_data["Account Number"] == logged_in_acct].count().iloc[0]
            if(count > transactions_count):
                return f"Too may transactions. {transactions_count} on file."
            else:
                return transaction_data[transaction_data["Account Number"] == logged_in_acct].head(count)
    else: return "Please Login"


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    """A function that takes a single prompt and sends it to the OpenAI model.
    Args:
    prompt: The message to send to the model
    model: default("gpt-3.5-turbo") - the openAI model to retrive a response from.

    Returns:
    String - OpenAI Response content
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages,tools,tool_choice=None, model="gpt-3.5-turbo", temperature=0):
    """A function that takes a series of messages and sends it to the OpenAI model.
     Tools are also specified, which enables the models function calling abilities.
     The series allows the model to continue to understand the previous context of the 
     entire conversation as opposed to a single prompt.

    Args:
        messages: The number of recent transactions to retrieve.  e.g. 5
        tools: tool definitions available to the model for function calling
        tool_choice(optional): choose a specific function for the model to call
        temperature: Default(0) Controls the randomness of the models output. 

    Returns:
        Transaction List - Pandas Data Frame
    """
        
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response



def add_to_context(role, content, tool_call_id=None, name=None):
    """Appends the chosen content to the OpenAI context list. The context
    list is passed to the OpenAI api in order for the model to remember
    its interactions with the user.

    Args:
        role: The role of the context.  e.g. user,assistant,tool
        content: The text content to send to the model.
        tool_call_id(optional): The id of the tool call.  This is necessary
            when we respond after the model utilizes function calling
        name: The name of the method utilized in function calling. e.g. account_balance_inquiry 

    Returns:
        None
    """
    entry = {'role': role, 'content': content}
    if tool_call_id is not None and name is not None:
        entry.update({'tool_call_id': tool_call_id, 'name': name})
    context.append(entry)
    


def process_openai_responses(response,temperature):
    """If the model, needs to call a function, this method preprocesses the model response
    before adding it the the context and the UI.  For security reasons, we send back
    placeholders to the models context and replace those placeholders with user information
    to send back to the UI.  For general banking info such as interest rates, hours, etc. we
    call the bank info method, which does a semantic search on the banks knowledge base.

    Args:
        response - The response from the OpenAI API after the User Submits A Request

    Returns:
        String - Final Response to send to the UI
    """
    if("tool_calls" in response.choices[0].message):
        runs = 1;
        #retrieve function with parameters from api response
        function = response.choices[0].message.tool_calls[0].function
        function_name = function.name

        tool_id = response.choices[0].message.tool_calls[0].id
        params = json.loads(function.arguments)
        
        #call function and save output
        chosen_function = functions_map.get(function_name)
        
        if(function_name == "bank_info"):
            function_output = chosen_function(prompt)
            add_to_context("tool",function_output,tool_id,function_name)
            chat_response = get_completion_from_messages(context,tools,temperature=temperature)
            add_to_context("assistant",chat_response.choices[0].message.content)
            final_response = chat_response.choices[0].message.content
            return final_response
        else:
            function_output = chosen_function(**params)
            placeholder = ""
            if(function_name == "account_balance_inquiry"):
                placeholder = "{balance}"
            else: placeholder = "{transactions}"

            #add tool message to context
            add_to_context("tool",placeholder,tool_id,function_name)

            #prompt engineering. Ensure's a consistent output of placeholders from the model
            add_to_context("system","Format your response in such a way that any balances or transactions given\
                            to the customer are placed at the end of the sentence and replaced with {balance} for balances and {transactions}\
                            for transactions.  Examples: I hope that satisfies your curiosity. Is there anything else you'd like to know, or are you done wasting my time?\
                            Let's see... After a painful search through our ancient transaction records, I found the last 4 transactions \
                            in your account: {transactions}")

            #get the new response and add it to the context
            chat_response = get_completion_from_messages(context,tools,temperature=temperature)
            add_to_context("assistant",chat_response.choices[0].message.content)
            final_response = chat_response.choices[0].message.content

            #replace placeholders with private data
            if(function_name == "account_balance_inquiry"):
                final_response = final_response.replace(placeholder,f'${function_output}')
            else:
                transaction_details = ""
                for index, transaction in function_output.iterrows():
                    # Format the transaction details string
                    transaction_details = f"\n{transaction_details} {transaction['Merchant Name']} - ${transaction['Transaction Amount']}\n"
                    # Replace the specific transaction placeholder with details
                placeholder = "{transactions}"
                final_response = final_response.replace(placeholder, transaction_details, 1)

            return final_response
    else: return response.choices[0].message.content

#Map function names to method definitions
functions_map = {
    "account_balance_inquiry": account_balance_inquiry,
    "get_transaction_history": get_transaction_history,
    "bank_info": bank_info,
}

prompt = ''

def collect_messages(user_input,temperature=0):
    """Collects User Prompts from the UI, processes them, and outputs the model responses to the UI.

    Args:
        None

    Returns:
       Panel Columns

    """
    global prompt
    prompt = user_input
    #inp.value = ''
    add_to_context('user',prompt)

    response = get_completion_from_messages(context, tools, model="gpt-3.5-turbo", temperature=temperature, tool_choice=None)
    context.append(response.choices[0].message)
    chat_response = process_openai_responses(response,temperature)
    add_to_context('assistant',chat_response)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": chat_response})

#######################################
#Semantic Search - Knowledge Base #
#######################################

bank_info = [
    {"topic": "Interest Rates", "info": "Find out about our competitive interest rates for savings, checking, and loan accounts."},
    {"topic": "Savings Account Interest", "info": "Our savings account offers a 0.5% APR."},
    {"topic": "Checking Account Interest", "info": "Our checking accounts have a 0.1% APR."},
    {"topic": "Loan Interest Rates", "info": "Loan rates start at 3.0% APR for qualified borrowers."},
    {"topic": "Certificate of Deposit", "info": "Earn up to 1.5% APR with our CD accounts, available in terms of 1 to 5 years."},
    {"topic": "Mortgage Rates", "info": "We offer mortgage rates as low as 3.5% APR for a 30-year fixed mortgage."},
    {"topic": "Branch Locations", "info": "Find branches near you for in-person banking services."},
    {"topic": "Downtown Branch", "info": "123 Main St, Downtown City. Open Mon-Fri, 9 AM - 5 PM."},
    {"topic": "Suburban Branch", "info": "456 Suburb Ln, Suburb Town. Open Mon-Sat, 10 AM - 4 PM."},
    {"topic": "Online Banking", "info": "Access your account 24/7 with our secure online banking platform."},
    {"topic": "Mobile Banking", "info": "Manage your account on the go with our mobile app, available for iOS and Android."},
    {"topic": "Account Opening", "info": "Start banking with us today. Open an account online or at any branch with just $100."},
    {"topic": "Overdraft Protection", "info": "Protect yourself from overdraft fees with our overdraft protection services."},
    {"topic": "Personal Loans", "info": "Competitive rates for personal loans to help you achieve your financial goals."},
    {"topic": "Auto Loans", "info": "Finance your next vehicle with our low-rate auto loans."},
    {"topic": "Financial Advice", "info": "Speak with our financial advisors to plan your financial future."},
    {"topic": "Investment Services", "info": "Explore investment opportunities with our wealth management services."},
    {"topic": "Safety and Security", "info": "Your security is our top priority. Learn more about how we protect your financial information."},
    {"topic": "Customer Support", "info": "Contact our support team for help with any banking needs. Available 24/7."},
]

topic_text = [item['topic'] for item in bank_info]

embeddings_response = openai.Embedding.create(
model='text-embedding-3-small',
    input=topic_text
)

#convert the response to a dictionary
response_dict = dict(embeddings_response)
#rint(response_dict)

for i, topic in enumerate(bank_info):
    topic['embedding'] = response_dict['data'][i]['embedding']

# User input
user_input = st.text_input("Type your message here...", key="user_input")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I assist you today?"}]

# Button to send message
if st.button('Send'):
    if user_input:  # Check if input is not empty
        chat_response = collect_messages(user_input)
        # Update UI with the response
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": chat_response})
        # Clear input box (This part is handled automatically in Streamlit)
    else:
        st.error("Please enter a message.")

# Display existing chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    if role == "user":
        st.text_area("You", value=content, height=100, disabled=True)
    else:  # role == "assistant"
        st.text_area("Chatbot", value=content, height=100, disabled=True, key=message["content"][:10])