from pathlib import Path
import pickle
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "spam_pipeline.pkl"

st.set_page_config(page_title="MailGuard ML", page_icon="✉", layout="wide")
st.markdown("<style>.stApp{background:#070b10;color:#eef3f8}.block-container{max-width:1200px;padding-top:2rem}.hero h1{font-size:3rem;letter-spacing:-.05em}.hero p{color:#91a0ae}</style>", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>MailGuard ML</h1><p>Email spam detection and classification.</p></div>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


if not MODEL_PATH.exists():
    st.warning("No trained classifier is available.")
    st.code("python train.py")
    st.stop()

model = load_model()
message = st.text_area("Message", height=240, placeholder="Paste an email or message")
if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.warning("Enter a message before inference.")
    else:
        label = "SPAM" if int(model.predict([message])[0]) == 1 else "HAM"
        if label == "SPAM":
            st.error(f"Classification · {label}")
        else:
            st.success(f"Classification · {label}")
        if hasattr(model, "decision_function"):
            st.metric("Decision score", f"{float(model.decision_function([message])[0]):.3f}")
