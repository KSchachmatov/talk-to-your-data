# --------------------------------------------------------------
# Import libraries
# --------------------------------------------------------------
import re
import openai
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import tool
from plotly.io import from_json
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

openai.api_key = st.secrets["OPENAI_API_KEY"]
# --------------------------------------------------------------
# Function to extract Python code from a text block 
# --------------------------------------------------------------
# This function uses a regular expression to find Python code blocks in the text.
def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]
    
# --------------------------------------------------------------
# Tool to Plot Charts
# --------------------------------------------------------------
@tool
def plot_scatter(data: str) -> str:
    """
    Custom tool to render scatter plots. The input 'data' should include arguments
    like x and y column names and the dataframe reference (df).
    E.g., 'df.plot.scatter(x="Age", y="Survived")'
    """
    try:
        # Use an `exec` approach to execute the code (data contains the plotting instruction)
        exec(data)
        
        # Render the plot in Streamlit
        st.pyplot(plt)
        
        # Optionally, clear the figure after rendering
        plt.clf()
        
        return "Scatter plot has been successfully plotted."
    except Exception as e:
        return f"Error occurred while plotting: {e}"



# --------------------------------------------------------------
# App UI
# --------------------------------------------------------------
st.title("🤖 Talk to Your Data")

# Dropdown to select a dataframe
option = st.selectbox(
    "Please select a dataframe you want to explore:",
    ("Titanic", "DS_Salaries"),
    index=None,
    placeholder="Select a dataframe...",
)

# --------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Titanic":
        return pd.read_csv("../data/titanic.csv")
    elif dataset_name == "DS_Salaries":
        return pd.read_csv("../data/ds_salaries.csv")
    else:
        return pd.DataFrame()

df = load_data(option)
st.write(f"You selected: **{option}**")

# Display the first few rows of the dataset as a preview
if not df.empty:
    st.dataframe(df.head())
else:
    st.warning("The selected dataset is empty or could not be loaded.")

# --------------------------------------------------------------
# Initialize and reset chat if a new dataframe is selected
# --------------------------------------------------------------
# Initialize Language Model
llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0.0)
tools = [plot_scatter]
# Session state to store agent and chat messages
if "agent" not in st.session_state or "selected_dataframe" not in st.session_state or st.session_state.selected_dataframe != option:
    # Reset agent and chat history when a new dataframe is selected
    st.session_state.agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True, 
        extra_tools=tools,
        # return_intermediate_steps=True,
        # extra_tools=tools
        )
    st.session_state.selected_dataframe = option
    st.session_state.messages = []

# --------------------------------------------------------------
# Display chat history
# --------------------------------------------------------------
# st.write("### Chat History")

# Create a container for the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------------
# Accept user input and handle response
# --------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the dataset:"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respond using the agent
    assistant_reply = None

    with st.chat_message("assistant"):
        assistant_reply = st.session_state.agent.run(prompt)
        # if "plot" in st.session_state.messages[-1]["content"].lower():
        #     code = assistant_reply["intermediate_steps"][0][0].tool_input
        #     # code = code.replace("fig.show()", "")
        #     code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
        #     st.write(f"```{code}")
        #     exec(code)
            
        #     # st.plotly_chart(assistant_reply)
        # else:
        st.markdown(assistant_reply)

    # Add assistant reply to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# # --------------------------------------------------------------
# # Perform basic data exploration
# # --------------------------------------------------------------

# agent.run("how many rows and columns are there in the dataset?")

# agent.run("are there any missing values?")

# agent.run("what are the columns?")

# agent.run("how many categories are in each column?")
# agent.run("plot age against survival")

# # --------------------------------------------------------------
# # Perform multiple-steps data exploration
# # --------------------------------------------------------------

# agent.run("which are the top 5 jobs that have the highest median salary?")

# agent.run("what is the percentage of data scientists who are working full time?")

# agent.run("which company location has the most employees working remotely?")

# agent.run("what is the most frequent job position for senior-level employees?")

# agent.run(
#     "what are the categories of company size? What is the proportion of employees they have? What is the total salary they pay for their employees?"
# )
# agent.run(
#     "get median salaries of senior-level data scientists for each company size and plot them in a bar plot."
# )

# # --------------------------------------------------------------
# # Initialize an agent with multiple dataframes
# # --------------------------------------------------------------

# df_2022 = df[df["work_year"] == 2022]
# df_2023 = df[df["work_year"] == 2023]

# agent = create_pandas_dataframe_agent(llm, [df_2022, df_2023], verbose=True)

# # --------------------------------------------------------------
# # Perform basic & multiple-steps data exploration for both dataframes
# # --------------------------------------------------------------

# agent.run("how many rows and columns are there for each dataframe?")

# agent.run(
#     "what are the differences in median salary for data scientists among the dataframes?"
# )
# agent.run(
#     "how many people were hired for each of the dataframe? what are the percentages of experience levels?"
# )
# agent.run(
#     "what is the median salary of senior data scientists for df2, given there is a 10% increment?"
# )
