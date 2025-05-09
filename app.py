import streamlit as st
import os
import tempfile
import base64
import requests
import fitz
from PIL import Image
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

@st.cache_resource
def load_models():
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return text_model, clip_model, clip_processor

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_ollama(model, prompt, images=None, system=None):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    if images:
        data["images"] = images
    if system:
        data["system"] = system
    response = requests.post(url, json=data)
    return response.json().get("response", "") if response.ok else "[Error in response]"

def extract_content(pdf_path, text_embedder, clip_model, clip_processor):
    doc = fitz.open(pdf_path)
    text_chunks, text_embeds, image_paths, image_embeds = [], [], [], []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            text_chunks.append((i, text))
            embed = text_embedder.encode(text, normalize_embeddings=True)
            text_embeds.append(embed)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_bytes)
                path = tmp.name
            image_paths.append({"path": path, "page": i + 1})
            try:
                image = Image.open(path).convert("RGB")
                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    emb = clip_model.get_image_features(**inputs)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                    image_embeds.append(emb.squeeze().numpy())
            except Exception as e:
                print(f"Error processing image on page {i+1}, index {img_index}: {e}")
            finally:
                if os.path.exists(path):
                    os.remove(path)
    return text_chunks, text_embeds, image_paths, image_embeds

def retrieve_relevant(query, text_embedder, clip_model, clip_processor, text_chunks, text_embeddings, image_paths, image_embeddings, top_k=3):
    relevant_texts = []
    relevant_images = []
    if text_embeddings:
        query_vec_text = text_embedder.encode(query, normalize_embeddings=True)
        text_embs = np.array(text_embeddings)
        if len(text_embs.shape) == 1:
            text_embs = text_embs.reshape(1, -1)
        elif len(text_embs.shape) > 2:
            text_embs = text_embs.reshape(-1, text_embs.shape[-1])
        if text_embs.shape[0] > 0:
            text_index = faiss.IndexFlatIP(text_embs.shape[1])
            text_index.add(text_embs)
            D_text, I_text = text_index.search(np.array([query_vec_text]), top_k)
            relevant_texts = [text_chunks[i][1] for i in I_text[0]]
    if image_embeddings:
        query_vec_image_inputs = clip_processor(text=[query], return_tensors="pt")
        with torch.no_grad():
            query_vec_image = clip_model.get_text_features(**query_vec_image_inputs).squeeze().numpy()
            query_vec_image = query_vec_image / np.linalg.norm(query_vec_image)
        image_embs = np.array(image_embeddings)
        if len(image_embs) > 0:
            if len(image_embs.shape) == 1:
                image_embs = image_embs.reshape(1, -1)
            elif len(image_embs.shape) > 2:
                image_embs = image_embs.reshape(-1, image_embs.shape[-1])
            embedding_dimension = image_embs.shape[1]
            image_index = faiss.IndexFlatIP(embedding_dimension)
            image_index.add(image_embs)
            D_image, I_image = image_index.search(np.array([query_vec_image]), min(top_k, image_embs.shape[0]))
            relevant_images = [image_paths[i] for i in I_image[0]]
    return relevant_texts, relevant_images

def process_query(query, text_embedder, clip_model, clip_processor, text_chunks, text_embeddings, image_paths, image_embeddings):
    retrieved_texts, retrieved_images = retrieve_relevant(query, text_embedder, clip_model, clip_processor, text_chunks, text_embeddings, image_paths, image_embeddings)
    used_imgs = retrieved_images
    joined_texts = "\n\n".join(retrieved_texts)
    base64_images = [encode_image(img["path"]) for img in retrieved_images if os.path.exists(img["path"])]
    prompt = f"""Question: {query}\n\nDocument Text Content:\n{joined_texts}\n\nConsider the attached images as context. Use both text and image content."""
    system_msg = "You are a helpful assistant that answers questions about documents using provided text and images."
    response = query_ollama("llava", prompt, images=base64_images, system=system_msg)
    return response, used_imgs

st.set_page_config(page_title="Multimodal RAG", page_icon="🔥", layout="wide")
st.title("Multimodal RAG App")
st.markdown("Upload a PDF and ask questions about its content")

text_embedder, clip_model, clip_processor = load_models()

if "documents" not in st.session_state:
    st.session_state["documents"] = []
if "text_chunks" not in st.session_state:
    st.session_state["text_chunks"] = []
if "text_embeddings" not in st.session_state:
    st.session_state["text_embeddings"] = []
if "image_paths" not in st.session_state:
    st.session_state["image_paths"] = []
if "image_embeddings" not in st.session_state:
    st.session_state["image_embeddings"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

with st.sidebar:
    st.header("Upload Document")
    file = st.file_uploader("Upload PDF", type="pdf")
    if file and st.button("Process"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            path = tmp.name
        with st.spinner("Extracting content and computing embeddings..."):
            tc, te, ip, ie = extract_content(path, text_embedder, clip_model, clip_processor)
            st.session_state.text_chunks = tc
            st.session_state.text_embeddings = te
            st.session_state.image_paths = ip
            st.session_state.image_embeddings = ie
            st.session_state.documents = [file]
            st.success("Document processed.")

st.header("Sawal poocho")
query = st.text_input("Your question")
if query and st.button("Submit"):
    if st.session_state.text_embeddings and st.session_state.image_embeddings:
        with st.spinner("Generating response..."):
            response, used_imgs = process_query(query, text_embedder, clip_model, clip_processor, st.session_state.text_chunks, st.session_state.text_embeddings, st.session_state.image_paths, st.session_state.image_embeddings)
            found = False
            for item in st.session_state.chat_history:
                if item['query'] == query:
                    item['responses'].append({'response': response, 'images': used_imgs, 'feedback': None, 'regenerated': False})
                    found = True
                    break
            if not found:
                st.session_state.chat_history.append({'query': query, 'responses': [{'response': response, 'images': used_imgs, 'feedback': None, 'regenerated': False}]})
    else:
        st.warning("Please upload and process a document first.")

for i, exch in enumerate(st.session_state.chat_history):
    st.subheader(f"Q: {exch['query']}")
    for response_index, response_data in enumerate(exch['responses']):
        st.write(response_data['response'])
        if response_data.get("images"):
            st.subheader("Images Used:")
            cols = st.columns(min(3, len(response_data["images"])))
            for j, img in enumerate(response_data["images"]):
                with cols[j % len(cols)]:
                    if os.path.exists(img["path"]):
                        st.image(img["path"], caption=f"Page {img['page']}", use_column_width=True)
        feedback_key = f"fb_{i}_{response_index}"
        feedback = st.radio(f"Was this helpful?", ["👍", "👎"], horizontal=True, key=feedback_key, index=0)

        if feedback == "👎" and not response_data.get('regenerated', False) and st.session_state.get(f"regenerate_{i}_{response_index}", False) is False:
            with st.spinner("Regenerating response..."):
                new_response, new_used_imgs = process_query(exch['query'], text_embedder, clip_model, clip_processor, st.session_state.text_chunks, st.session_state.text_embeddings, st.session_state.image_paths, st.session_state.image_embeddings)
                exch['responses'].append({'response': new_response, 'images': new_used_imgs, 'feedback': None, 'regenerated': True})
                st.session_state[f"regenerate_{i}_{response_index}"] = True
                st.rerun()

        if feedback:
            exch['responses'][response_index]['feedback'] = feedback
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{exch['query']}\t{response_data['response']}\t{feedback}\n")
            st.success("Thank you for your feedback!", icon="✅")

    st.divider()

with st.expander("Document Info"):
    if st.session_state.documents:
        st.write("Uploaded:", st.session_state.documents[0].name)
        st.write("Total Images:", len(st.session_state.image_paths))
        st.write("Total Text Chunks:", len(st.session_state.text_chunks))
