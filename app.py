import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from src.recommender import load_books, BookRecommender
from experiments import experiment_01, experiment_02, experiment_03, experiment_04, experiment_05, experiment_06, experiment_07, experiment_08, experiment_09, experiment_10

st.set_page_config(page_title="PageTurn | Intelligent Book Discovery", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root{--bg:#0b1020;--panel:#141b2e;--panel2:#1a2440;--line:#2b3654;--gold:#e8b978;--mint:#9fd6c1;--muted:#aeb9d0}
.stApp{background:radial-gradient(ellipse at 0% 0%,#202848 0%,#0d1324 42%,#090d18 100%);font-family:'DM Sans',sans-serif;color:#f4f5fa}
[data-testid="stSidebar"]{background:#101729;border-right:1px solid #27324d}
.block-container{max-width:1450px;padding-top:1.25rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Playfair Display',serif!important;color:#fff7e9!important;letter-spacing:-.025em}
p,label,.stMarkdown{color:#d4daea}
.hero{padding:2.6rem;border:1px solid #364363;border-radius:28px;background:linear-gradient(120deg,rgba(33,45,75,.97),rgba(19,31,49,.98));box-shadow:0 25px 80px #0005;margin-bottom:1.3rem;position:relative;overflow:hidden}
.hero:after{content:'✦';position:absolute;right:6%;top:-25%;font-size:13rem;color:#e8b97815}
.eyebrow{font-size:.72rem;text-transform:uppercase;letter-spacing:.2em;color:var(--gold);font-weight:700}
.hero h1{font-size:3.3rem;line-height:1.06;margin:.6rem 0}
.hero p{font-size:1.03rem;max-width:750px;line-height:1.7;color:#bdc8df}
.tag{display:inline-block;padding:.35rem .7rem;border:1px solid #485675;border-radius:999px;color:#e9cda2;font-size:.75rem;margin:.25rem .3rem 0 0}
.kpi{padding:1.1rem 1.25rem;background:linear-gradient(145deg,#1b2742,#131b2d);border:1px solid #2d3b5c;border-radius:17px;min-height:110px}
.kpi-num{font-size:1.8rem;font-weight:700;color:var(--gold)}.kpi-label{font-size:.82rem;color:#aeb9d0}
.book{padding:1.05rem 1.15rem;background:linear-gradient(145deg,#19243c,#131a2b);border:1px solid #2b3958;border-radius:16px;margin:.4rem 0 .8rem}
.book:hover{border-color:#e8b978}
.booktitle{font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:600;color:#fff6e8}
.author{color:#e8b978;font-size:.88rem;margin:.25rem 0 .55rem}
.genre{font-size:.72rem;background:#203b38;color:#9fd6c1;border-radius:999px;padding:.25rem .6rem;display:inline-block}
.muted{color:#aeb9d0;font-size:.9rem;line-height:1.55}
div.stButton>button{background:linear-gradient(100deg,#e8b978,#f3d09a);color:#172033;border:0;border-radius:11px;font-weight:700;min-height:2.6rem}
div.stButton>button:hover{background:#f5d9a9;color:#172033;border:0;box-shadow:0 6px 24px #e8b97825}
.stTextInput input,.stTextArea textarea,.stSelectbox [data-baseweb=select],.stNumberInput input{background:#10182a!important;color:#f4f5fa!important;border-color:#344260!important;border-radius:11px!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#e8b978;border-bottom-color:#e8b978}
hr{border-color:#2a3551}footer{visibility:hidden}#MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_books(): return load_books()
@st.cache_resource
def get_engine(): return BookRecommender(get_books())
books=get_books()
engine=get_engine()
genres=["All genres"]+sorted(books["genre"].dropna().astype(str).unique().tolist())
authors=["All authors"]+sorted(books["authors"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.markdown("# 📚 PageTurn")
    st.caption("INTELLIGENT BOOK DISCOVERY")
    st.divider()
    page=st.radio("WORKSPACE",["Command Center","Recommendation Studio","Explore Catalog","Reading Profile","Collection Analytics","NLP Workbench"],label_visibility="visible")
    st.divider()
    st.markdown("#### Quick filters")
    global_genre=st.selectbox("Preferred genre",genres,key="global_genre")
    st.caption("Your local collection powers search and recommendations. Add your own CSV in Explore Catalog.")
    st.divider()
    st.caption("PAGETURN · PERSONALIZED DISCOVERY")

def book_card(row, score=True):
    title=str(row.get("title","Untitled")); author=str(row.get("authors","Unknown")); genre=str(row.get("genre","Other")); desc=str(row.get("description",""))
    st.markdown(f"""<div class="book"><span class="genre">{genre}</span><div class="booktitle" style="margin-top:.65rem">{title}</div><div class="author">by {author}</div><div class="muted">{desc}</div></div>""",unsafe_allow_html=True)
    if score and "similarity" in row.index:
        s=float(row["similarity"]); st.progress(max(0,min(1,s))); st.caption(f"Match score: {s*100:.1f}% · {row.get('rationale','Content similarity')}")
    if st.button("➕ Add to reading list",key="save_"+str(row.get("book_id",title))):
        saved=st.session_state.get("reading_list",[])
        if title not in saved: saved.append(title)
        st.session_state["reading_list"]=saved
        st.toast(f"Added {title} to your reading list")

def show_cards(df, score=True):
    if df.empty: st.info("No books matched. Try changing the query or filters."); return
    for _,row in df.iterrows(): book_card(row,score)

# HERO
st.markdown("""<div class="hero"><div class="eyebrow">A personal discovery engine for readers</div><h1>Find the book<br>you didn't know you needed.</h1><p>Explore a living catalogue, discover books by meaning and mood, build your reading list, and understand why each recommendation fits. Designed as an intelligent reading workspace—not just a search box.</p><span class="tag">Content intelligence</span><span class="tag">Personalized discovery</span><span class="tag">Reading-list workflow</span><span class="tag">Collection analytics</span></div>""",unsafe_allow_html=True)

if page=="Command Center":
    st.subheader("Command center")
    cols=st.columns(4)
    for c,num,label in zip(cols,[len(books),books["authors"].nunique(),books["genre"].nunique(),len(st.session_state.get("reading_list",[]))],["Books indexed","Authors","Genres","Saved to reading list"]):
        c.markdown(f"<div class='kpi'><div class='kpi-num'>{num}</div><div class='kpi-label'>{label}</div></div>",unsafe_allow_html=True)
    st.markdown("### Start with an intent")
    intent=st.text_input("Describe the story or ideas you want",placeholder="e.g. a clever mystery, isolated setting, psychological twists")
    a,b=st.columns([1,1])
    with a: num=st.select_slider("Recommendations",options=[3,5,8,10,15],value=5)
    with b: genre=st.selectbox("Limit to genre",genres,key="home_genre")
    if st.button("✨ Generate discovery shortlist",type="primary",use_container_width=True):
        st.session_state["shortlist"]=engine.recommend_for_preferences(intent,num,genre=genre)
    if "shortlist" in st.session_state: show_cards(st.session_state["shortlist"])
    st.markdown("### Popular entry points")
    c1,c2,c3=st.columns(3)
    for col,(name,query) in zip([c1,c2,c3],[("Mystery & suspense","mystery thriller crime detective"),("Big ideas","psychology history science"),("Other worlds","fantasy magic space planet")]):
        with col:
            st.markdown(f"<div class='kpi'><div class='eyebrow'>CURATED PATH</div><h3>{name}</h3><div class='muted'>Explore books that share themes and language.</div></div>",unsafe_allow_html=True)
            if st.button("Explore →",key="curated_"+name):
                st.session_state["shortlist"]=engine.recommend_for_preferences(query,5,genre=global_genre)
                st.rerun()

elif page=="Recommendation Studio":
    st.subheader("Recommendation studio")
    mode=st.radio("Recommendation strategy",["Similar to a book","Based on my interests","Blend both"],horizontal=True)
    left,right=st.columns([1.2,1])
    with left:
        title=st.selectbox("Seed book",books["title"].tolist())
        interests=st.text_area("Your interests, themes, or reading goals",placeholder="slow-burn mystery, scientific problem-solving, character growth")
    with right:
        n=st.slider("Results",3,15,8)
        exclude=st.multiselect("Exclude books already read",books["title"].tolist())
        g=st.selectbox("Genre constraint",genres,key="studio_genre")
    if st.button("Build my recommendation set",type="primary",use_container_width=True):
        if mode=="Similar to a book": out=engine.recommend(title,n,genre=g,exclude_read=exclude)
        elif mode=="Based on my interests": out=engine.recommend_for_preferences(interests,n,genre=g,exclude_titles=exclude)
        else:
            a=engine.recommend(title,n*2,genre=g,exclude_read=exclude)
            b=engine.recommend_for_preferences(interests,n*2,genre=g,exclude_titles=exclude)
            combined=pd.concat([a,b]).drop_duplicates("title")
            if "similarity" in combined: combined=combined.sort_values("similarity",ascending=False)
            out=combined.head(n)
        st.session_state["studio_results"]=out
    if "studio_results" in st.session_state: show_cards(st.session_state["studio_results"])
    if "studio_results" in st.session_state and not st.session_state["studio_results"].empty:
        csv=st.session_state["studio_results"].to_csv(index=False).encode("utf-8")
        st.download_button("Download recommendation shortlist (CSV)",csv,"my_recommendations.csv","text/csv")

elif page=="Explore Catalog":
    st.subheader("Explore the catalogue")
    with st.expander("Import your own book dataset (CSV)",expanded=False):
        st.write("Expected columns: title, authors (or author), genre (or category), description (or summary).")
        upload=st.file_uploader("Upload CSV",type=["csv"])
        if upload:
            try:
                user_df=pd.read_csv(upload).fillna("")
                st.success(f"Loaded preview with {len(user_df)} rows. This preview is session-only.")
                st.dataframe(user_df.head(100),use_container_width=True)
                st.download_button("Download uploaded CSV copy",upload.getvalue(),"uploaded_books.csv","text/csv")
            except Exception as e: st.error(f"Could not read CSV: {e}")
    q=st.text_input("Search across titles, authors, genres, and descriptions",placeholder="Type a keyword or phrase")
    c1,c2,c3=st.columns(3)
    with c1:g=st.selectbox("Genre",genres,key="catalog_genre")
    with c2:a=st.selectbox("Author",authors)
    with c3:s=st.selectbox("Sort by",["Relevance","Highest match","Title A–Z","Author A–Z"])
    out=engine.search(q,genre=g,author=a,limit=100,sort=s)
    st.caption(f"{len(out)} matching records")
    if not out.empty:
        st.dataframe(out[["title","authors","genre","description"]],use_container_width=True,hide_index=True)
        st.download_button("Export current results CSV",out[["title","authors","genre","description"]].to_csv(index=False).encode(), "catalog_results.csv","text/csv")
    else: st.info("Try a different search term.")

elif page=="Reading Profile":
    st.subheader("Your reading profile")
    st.write("Keep a lightweight local reading list and track what you want to read next.")
    if "reading_list" not in st.session_state: st.session_state["reading_list"]=[]
    titles=books["title"].tolist()
    selected=st.multiselect("Add books to your list",titles)
    if st.button("Save selected books"):
        st.session_state["reading_list"]=list(dict.fromkeys(st.session_state["reading_list"]+selected))
        st.toast("Reading list updated")
    status=st.radio("Show",["All saved books","To read","Finished"],horizontal=True)
    if "finished_books" not in st.session_state: st.session_state["finished_books"]=[]
    saved=st.session_state["reading_list"]
    if status=="Finished": saved_view=[x for x in saved if x in st.session_state["finished_books"]]
    elif status=="To read": saved_view=[x for x in saved if x not in st.session_state["finished_books"]]
    else: saved_view=saved
    st.markdown(f"### {len(saved_view)} book(s)")
    for title in saved_view:
        row=books[books["title"]==title].iloc[0]
        c1,c2=st.columns([4,1])
        with c1: st.markdown(f"**{title}** · {row['authors']} · {row['genre']}")
        with c2:
            if title not in st.session_state["finished_books"]:
                if st.button("Mark finished",key="finished_"+title):
                    st.session_state["finished_books"].append(title); st.rerun()
            else: st.caption("✓ Finished")
            if st.button("Remove",key="remove_"+title):
                st.session_state["reading_list"].remove(title); st.rerun()
    if saved:
        export=pd.DataFrame({"title":saved,"status":["Finished" if x in st.session_state["finished_books"] else "To read" for x in saved]})
        st.download_button("Export reading list",export.to_csv(index=False).encode(),"reading_list.csv","text/csv")

elif page=="Collection Analytics":
    st.subheader("Collection intelligence")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Catalogue by genre")
        counts=books["genre"].value_counts()
        fig,ax=plt.subplots(figsize=(7,4),facecolor="#141b2e"); ax.set_facecolor("#141b2e")
        counts.plot(kind="bar",ax=ax,color="#e8b978"); ax.tick_params(axis="x",rotation=35,colors="#d4daea"); ax.tick_params(axis="y",colors="#d4daea")
        ax.set_ylabel("Books",color="#d4daea"); ax.spines[["top","right"]].set_visible(False); st.pyplot(fig)
    with c2:
        st.markdown("#### Author catalogue")
        counts=books["authors"].value_counts().head(10).sort_values()
        fig,ax=plt.subplots(figsize=(7,4),facecolor="#141b2e"); ax.set_facecolor("#141b2e")
        counts.plot(kind="barh",ax=ax,color="#9fd6c1"); ax.tick_params(colors="#d4daea"); ax.set_xlabel("Books",color="#d4daea"); ax.spines[["top","right"]].set_visible(False); st.pyplot(fig)
    st.markdown("#### Collection data quality")
    quality=pd.DataFrame({"Field":["title","authors","genre","description"],"Missing values":[int((books[c].astype(str).str.strip()=="").sum()) for c in ["title","authors","genre","description"]],"Unique values":[books[c].nunique() for c in ["title","authors","genre","description"]]})
    st.dataframe(quality,use_container_width=True,hide_index=True)
    st.download_button("Download catalogue data",books[["title","authors","genre","description"]].to_csv(index=False).encode(),"book_catalog.csv","text/csv")

elif page=="NLP Workbench":
    st.subheader("Text analysis workbench")
    st.caption("Interactive text analysis tools for experimenting with natural-language features.")
    exp=st.selectbox("Choose a tool",["Tokenization and filtration","Stemming and lemmatization","Morphological analysis","N-gram language model","POS tagging","Chunking","Named entity recognition","Document similarity","Word cloud","Project use case"])
    text=st.text_area("Input text","The young reader is exploring a mysterious science fiction story.",height=110)
    if exp=="Document similarity": second=st.text_area("Second document","A reader enjoys science fiction and mystery stories.")
    if exp=="N-gram language model":
        n=st.slider("N",2,4,2); context=st.text_input("Prediction context","the")
    if st.button("Analyze text",type="primary"):
        if exp=="Tokenization and filtration": st.json(experiment_02.run(text))
        elif exp=="Stemming and lemmatization": st.json(experiment_03.run(text))
        elif exp=="Morphological analysis": st.json(experiment_04.run(text))
        elif exp=="N-gram language model": st.json(experiment_05.run(text,n=n,context=context,seed=context,num_words=10))
        elif exp=="POS tagging": st.write(experiment_06.run(text))
        elif exp=="Chunking": st.json(experiment_07.run(text))
        elif exp=="Named entity recognition": st.json(experiment_08.run(text))
        elif exp=="Document similarity": st.json(experiment_09.run(text,second))
        elif exp=="Word cloud":
            result=experiment_10.run(text); st.json(result["frequencies"])
            if result["frequencies"]:
                cloud=WordCloud(width=1000,height=450,background_color="#141b2e",stopwords=set(STOPWORDS),colormap="Set2").generate(result["cleaned_text"])
                fig,ax=plt.subplots(figsize=(12,5),facecolor="#141b2e"); ax.imshow(cloud,interpolation="bilinear"); ax.axis("off"); st.pyplot(fig)
        else: st.json(experiment_01.get_project_use_case())

st.markdown("<hr><div style='text-align:center;color:#7f8aa5;font-size:.78rem;padding:1rem'>PAGETURN · INTELLIGENT BOOK DISCOVERY WORKSPACE</div>",unsafe_allow_html=True)
