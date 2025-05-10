import streamlit as st
import os
import tempfile
import base64
import requests
import fitz  # for working with PDFs
from PIL import Image
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Cache the models so we don't reload every time
@st.cache_resource
def get_models():
    txt_model = SentenceTransformer("all-MiniLM-L6-v2")  # small and quick enough
    clip_mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return txt_model, clip_mdl, clip_proc

# Just reads image and base64s it for API
def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Not fancy error handling here — maybe wrap later
def ask_ollama(model_name, user_prompt, img_data=None, system_msg=None):
    payload = {
        "model": model_name,
        "prompt": user_prompt,
        "stream": False
    }
    if img_data:
        payload["images"] = img_data
    if system_msg:
        payload["system"] = system_msg
    try:
        res = requests.post("http://localhost:11434/api/generate", json=payload)
        return res.json().get("response", "") if res.ok else "[Error from Ollama]"
    except Exception as e:
        return f"[Request failed: {str(e)}]"

# Go through each page of PDF and get the text + images
def parse_pdf(pdf_file_path, embedder, clip_net, processor):
    pdf = fitz.open(pdf_file_path)
    chunks, chunk_vecs = [], []
    img_meta, img_vecs = [], []

    for pg_num, pg in enumerate(pdf):
        txt = pg.get_text().strip()
        if txt:
            chunks.append((pg_num, txt))
            vec = embedder.encode(txt, normalize_embeddings=True)
            chunk_vecs.append(vec)

        for i, im in enumerate(pg.get_images(full=True)):
            xref = im[0]
            raw_img = pdf.extract_image(xref)["image"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(raw_img)
                img_path = tmp_img.name

            img_meta.append({"path": img_path, "page": pg_num + 1})
            try:
                img_obj = Image.open(img_path).convert("RGB")
                inputs = processor(images=img_obj, return_tensors="pt")
                with torch.no_grad():
                    out = clip_net.get_image_features(**inputs)
                    norm_out = torch.nn.functional.normalize(out, p=2, dim=-1)
                    img_vecs.append(norm_out.squeeze().numpy())
            except Exception as ex:
                print(f"Img error on page {pg_num + 1} idx {i}: {ex}")
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)

    return chunks, chunk_vecs, img_meta, img_vecs

# Helper to fetch relevant items using FAISS
def find_matches(query_text, embedder, clip_net, processor, all_chunks, chunk_vecs, imgs, img_vecs, top_k=3):
    found_texts = []
    found_imgs = []

    if chunk_vecs:
        qv_text = embedder.encode(query_text, normalize_embeddings=True)
        text_arr = np.array(chunk_vecs)
        if len(text_arr.shape) == 1:
            text_arr = text_arr.reshape(1, -1)
        elif len(text_arr.shape) > 2:
            text_arr = text_arr.reshape(-1, text_arr.shape[-1])
        if text_arr.size:
            index = faiss.IndexFlatIP(text_arr.shape[1])
            index.add(text_arr)
            _, idxs = index.search(np.array([qv_text]), top_k)
            found_texts = [all_chunks[i][1] for i in idxs[0]]

    if img_vecs:
        clip_inputs = processor(text=[query_text], return_tensors="pt")
        with torch.no_grad():
            img_query_vec = clip_net.get_text_features(**clip_inputs).squeeze().numpy()
            img_query_vec /= np.linalg.norm(img_query_vec)
        img_arr = np.array(img_vecs)
        if img_arr.size:
            if len(img_arr.shape) == 1:
                img_arr = img_arr.reshape(1, -1)
            elif len(img_arr.shape) > 2:
                img_arr = img_arr.reshape(-1, img_arr.shape[-1])
            faiss_idx = faiss.IndexFlatIP(img_arr.shape[1])
            faiss_idx.add(img_arr)
            _, matched = faiss_idx.search(np.array([img_query_vec]), min(top_k, len(img_arr)))
            found_imgs = [imgs[i] for i in matched[0]]

    return found_texts, found_imgs

# Main logic to process user input and call backend
def handle_query(q, embedder, clip_net, processor, txt_chunks, txt_embs, img_info, img_embs):
    matched_txts, matched_imgs = find_matches(q, embedder, clip_net, processor, txt_chunks, txt_embs, img_info, img_embs)
    # image paths are no longer available once deleted, so we re-encode
    b64_imgs = [image_to_base64(i["path"]) for i in matched_imgs if os.path.exists(i["path"])]

    prompt_text = f"""Question: {q}

Document Text Content:
{'\n\n'.join(matched_txts)}

Consider the attached images as context. Use both text and image content."""
    system_msg = "You are a helpful assistant that answers questions about documents using provided text and images."

    reply = ask_ollama("llava", prompt_text, images=b64_imgs, system_msg=system_msg)
    return reply, matched_imgs

# --- UI Starts Here ---

st.set_page_config(page_title="Multimodal RAG", page_icon="📄", layout="wide")
st.title("📚 Ask Your PDF")
st.markdown("Upload a PDF and then ask questions about what's inside. Works with both text and images.")

text_embedder, clip_model, clip_processor = get_models()

# Initialize session state if needed
for key in ["documents", "text_chunks", "text_embeddings", "image_paths", "image_embeddings", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = []

# Sidebar upload section
with st.sidebar:
    st.header("📤 Upload PDF")
    file = st.file_uploader("Choose your PDF", type="pdf")
    if file and st.button("📄 Process PDF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
            tmpf.write(file.getvalue())
            temp_path = tmpf.name

        with st.spinner("Extracting and embedding..."):
            t_chunks, t_embs, i_paths, i_embs = parse_pdf(temp_path, text_embedder, clip_model, clip_processor)
            st.session_state.text_chunks = t_chunks
            st.session_state.text_embeddings = t_embs
            st.session_state.image_paths = i_paths
            st.session_state.image_embeddings = i_embs
            st.session_state.documents = [file]
        st.success("Done! You can now ask questions.")

# Main interaction block
st.header("Ask Something")
user_query = st.text_input("Your question:")
if user_query and st.button("🔍 Get Answer"):
    if st.session_state.text_embeddings and st.session_state.image_embeddings:
        with st.spinner("Thinking..."):
            reply_text, images_used = handle_query(user_query, text_embedder, clip_model, clip_processor,
                                                   st.session_state.text_chunks, st.session_state.text_embeddings,
                                                   st.session_state.image_paths, st.session_state.image_embeddings)

            # Check if we already asked this one
            found = False
            for entry in st.session_state.chat_history:
                if entry['query'] == user_query:
                    entry['responses'].append({
                        'response': reply_text,
                        'images': images_used,
                        'feedback': None,
                        'regenerated': False
                    })
                    found = True
                    break
            if not found:
                st.session_state.chat_history.append({
                    'query': user_query,
                    'responses': [{
                        'response': reply_text,
                        'images': images_used,
                        'feedback': None,
                        'regenerated': False
                    }]
                })
    else:
        st.warning("You need to upload and process a document first!")

# Display past Q&A
for idx, chat in enumerate(st.session_state.chat_history):
    st.subheader(f"Q: {chat['query']}")
    for ridx, resp in enumerate(chat['responses']):
        st.write(resp['response'])
        if resp.get("images"):
            st.subheader("Images Used:")
            cols = st.columns(min(3, len(resp["images"])))
            for j, img in enumerate(resp["images"]):
                with cols[j % len(cols)]:
                    if os.path.exists(img["path"]):
                        st.image(img["path"], caption=f"Page {img['page']}", use_column_width=True)
        feedback_key = f"fb_{idx}_{ridx}"
        fb = st.radio("Helpful?", ["👍", "👎"], key=feedback_key, horizontal=True, index=0)

        if fb == "👎" and not resp.get("regenerated", False) and not st.session_state.get(f"regenerate_{idx}_{ridx}", False):
            with st.spinner("Trying again..."):
                retry_ans, retry_imgs = handle_query(chat['query'], text_embedder, clip_model, clip_processor,
                                                     st.session_state.text_chunks, st.session_state.text_embeddings,
                                                     st.session_state.image_paths, st.session_state.image_embeddings)
                chat['responses'].append({'response': retry_ans, 'images': retry_imgs, 'feedback': None, 'regenerated': True})
                st.session_state[f"regenerate_{idx}_{ridx}"] = True
                st.rerun()

        if fb:
            resp['feedback'] = fb
            with open("feedback_log.txt", "a", encoding="utf-8") as log:
                log.write(f"{chat['query']}\t{resp['response']}\t{fb}\n")
            st.success("Thanks for the feedback!", icon="✅")

    st.divider()

with st.expander("📝 PDF Info"):
    if st.session_state.documents:
        st.write("Uploaded:", st.session_state.documents[0].name)
        st.write("Text chunks:", len(st.session_state.text_chunks))
        st.write("Images found:", len(st.session_state.image_paths))
