####################################################
# GeoSELECT App
#
# Created for use with IMAGE 2023 abstract queries
# Chad Holmes, Steve Braun
# July 20, 2023
####################################################

import streamlit as st
import pandas as pd
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS

save_directory = Path(r"..\\data")

st.set_page_config(layout="wide")

use_openai = True

@st.cache_resource
def get_db():
    
    if use_openai:
        # Create a base embeddings api object
        apikey='YOUR API KEY'
        embedding_model = "text-embedding-ada-002"
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=apikey)
        
        # Load the vector db
        db = FAISS.load_local(save_directory / "faiss_image2023_docs_ada", embeddings)
        
    else:
        # Create a base embeddings api object
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        # Load the vector db
        db = FAISS.load_local(save_directory / "faiss_image2023_docs_mpnet", embeddings)

    return db

# load vector database
db = get_db()

# add a title to the webapp
llmbasis = ['openai' if use_openai else 'mpnet']
st.title("IMAGE 2023 - Presentation Finder ({})".format(llmbasis[0]))

question = st.text_input( 
    label="What question are you coming to IMAGE to answer for you or your organization?"
    )

num_sessions = st.slider(
    label="How many presentations would you like to consider?",
    min_value=1,
    max_value=25,
    value=5,
    )

def ask_question(question: str, num_sessions: int):
    # query the vector db
    query_result = db.similarity_search(question, k=num_sessions)
    data = []

    for doc in query_result:
        #print(doc.metadata)
        data.append(
            {'title': doc.metadata['title'],
             'authors': doc.metadata['authors'],
             'affiliations': doc.metadata['affiliations'],
             'bart_summary': doc.metadata['bart_summary'],
             'gpt_summary': doc.metadata['gpt_summary'],
             'abstract': doc.metadata['abstract'],
             'keywords': doc.metadata['keywords'],
             'session_name': doc.metadata['sessiontitle'],
             'session_type': doc.metadata['sessiontype'],
             'date': doc.metadata['date'],
             'time': doc.metadata['time'],
             'location': doc.metadata['sessionlocation']
            }
        )
    
    return data

def make_markdown_template(
    title: str = None,
    authors: str = None,
    affiliations: str=None,
    bart_summary: str=None,
    gpt_summary: str=None,
    abstract: str = None,
    keywords: str = None,
    session_name: str = None,
    date: str=None,
    time: str=None,
    location: str=None,
    **kwargs,
    ):

    template = (f"### Title: \n"
                f"#### {title} \n"
                f"### Authors: \n"
                f"#### {authors} \n"
                f"#### Affiliations: \n"
                f"{affiliations} \n"
                f"#### Summary (BART - 2019 LLM) \n"
                f"{bart_summary} \n"
                f"#### Summary (GPT3.5 - 2023 LLM) \n"
                f"{gpt_summary} \n"
                f"#### Abstract \n"
                f"{abstract} \n"
                f"#### Keywords \n"
                f"{keywords} \n"
                f"#### Session Name: \n" 
                f"{session_name} \n"
                f"#### Date: \n"
                f"{date} \n"
                f"#### Time: \n"
                f"{time} \n"
                f"#### Location: \n"
                f"{location} \n"
    )
    return template

data= []
if question:
    data = ask_question(question, num_sessions=num_sessions)

for d in data:
    with st.expander(f"{d['title']}"):
        st.markdown(make_markdown_template(**d))
