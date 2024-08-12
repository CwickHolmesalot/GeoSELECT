from pathlib import Path
import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# import pandas as pd

path = r'.'
dirpath = Path(path)

### helper functions
def ask_question(question: str, num_sessions: int, db: FAISS, threshold = 0.45):
    # query the vector db
    query_result = db.similarity_search_with_relevance_scores(query=question, k=num_sessions)
    data = []

    for tup in query_result:
      #print(tup[1])
      if tup[1] >= threshold:
        doc = tup[0]
        # print(doc.metadata)
        data.append({
             'title': doc.metadata['title'],
             'authors': doc.metadata['authors'],
             'keywords': doc.metadata['keywords'],
             'summary': doc.metadata['summary'],
             'session_name': doc.metadata['sessiontitle'],
             'session_type': doc.metadata['sessiontype'],
             'date': doc.metadata['date'],
             'time': doc.metadata['time'],
             'location': doc.metadata['sessionlocation'],
             'url': doc.metadata['url'],
             'similar': doc.metadata['similar_links'],
             'match':tup[1]
             })

    return data

def make_markdown_template(
    title: str = None,
    authors: str = None,
    keywords: str = None,
    summary: str = None,
    url: str = None,
    session_name: str = None,
    date: str = None,
    time: str = None,
    location: str = None,
    similar: str = None,
    **kwargs,
    ):

    template = (
                f"**Authors:**<br>"
                f" {authors}<br><br>"
                f"**Keywords:**<br>"
                f"{keywords}<br><br>"
                f"**Summary:**<br>"
                f"{summary}<br><br>"
                f"**Abstract:**<br>"
                f"{url}<br><br>"
                f"**Session Name:**<br>"
                f"{session_name}<br><br>"
                f"**Date:**<br>"
                f"{date}<br><br>"
                f"**Time:**<br>"
                f"{time}<br><br>"
                f"**Location:**<br>"
                f"{location}<br><br>"
                f"**Similar Abstracts:**<br>"
                f"{similar}<br><br>"
    )
    return template

st.set_page_config(layout="wide")

use_openai = True

@st.cache_resource
def get_db():

    # access to OpenAI libraries
    client = OpenAI(
      # This is the default and can be omitted
      api_key='INSERT API KEY HERE'
    )

    # specify embeddings model
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small',
                                        openai_api_key=client.api_key)

    # load vector database
    db = FAISS.load_local(dirpath / "faiss_image2024_db",
                          embeddings_model,
                          allow_dangerous_deserialization=True)
    return db

# load vector database
db = get_db()

# add a title to the webapp
st.title("IMAGE 2024 - Abstract Semantic Search Tool")


# Custom CSS for word wrapping
custom_css = """
<style>
    .wrapped-text {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
"""

st.markdown('<div class="wrapped-text">This site leverages AI technology for contextual matches between your query and the 1200+ abstracts at the 2024 IMAGE conference. For a good keyword/author search, visit the <a href="https://www.imageevent.org/technical-program-/full-schedule" target="_blank">IMAGE Technical Program</a>. </div><br>', unsafe_allow_html=True)

#st.markdown('<div class="wrapped-text">The match quality slider sets a similarity threshold between the query and the submitted abstract texts such that anything below the chosen threshold will not be returned.</div><br>', unsafe_allow_html=True)

#st.markdown('<div class="wrapped-text">The abstract count slider determines the maximum number of abstract matches to return, if matches above the quality threshold are found.</div><br>', unsafe_allow_html=True)

st.markdown("""Usage examples:
- Low context query: *FWI*
- Medium context query: *Exploration in Southeast Asia*
- High context query: *Use of Segment Anything AI model for image segmentation of seismic data*
""")

st.markdown('<br>',unsafe_allow_html=True)

question = st.text_input(
    label="What technical question or content are you looking for?  TIP: phrase like you're explaining to a friend, not as simplified keywords."
    )
    
match_thres = st.slider(
    label="Query match quality threshold determining which results to ignore (0.0=little to no relationship, 1.0=fantastic match)",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=0.30,
    )

num_sessions = st.slider(
    label="Maximum abstract count to return if matches above the quality threshold are found",
    min_value=1,
    max_value=25,
    value=5,
    )

data= []
if question:
    data = ask_question(question=question, num_sessions=num_sessions, db=db, threshold=match_thres)

for d in data:
    with st.expander(f"{d['title']} (Match Quality {d['match']:.0%})"):
        st.markdown(make_markdown_template(**d),unsafe_allow_html=True)