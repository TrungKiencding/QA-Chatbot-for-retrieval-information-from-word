import chainlit as cl
import chromadb
from FlagEmbedding import BGEM3FlagModel
import torch
from chainlit.types import AskFileResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=100)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # setting model retrieval

# setting model generation
model_gen = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_gen.eval()

client = chromadb.Client()
collection = client.create_collection("collection")


def process_file(file: AskFileResponse):

    file_loader = UnstructuredWordDocumentLoader

    loader = file_loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    return docs


def get_vector_db(file: AskFileResponse):
    docs = process_file(file)
    cl.user_session.set("docs", docs)
    docs_embedding = model.encode(docs)['dense_vecs']
    for i, doc in enumerate(docs):
        collection.add(
            documents=[doc],  # Tài liệu
            embeddings=[docs_embedding[i]],  # Embedding tương ứng
            metadatas=[{"source": f"doc_{i}"}],
            ids=[f"doc_{i}"]  # ID cho mỗi tài liệu
        )
    return collection


def generate_answer(prompt: str):
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, padding=True, max_length=512)

    # Sinh câu trả lời từ GPT-2
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=200,
                                 num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

    # Giải mã kết quả
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


welcome_message = """Welcome to the Word QA! To get started:
1. Upload a Word file
2. Aask a question about the file
"""


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["appilication/msword"],
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]

    msg = cl.Message(content=f"Processing '{file.name}'...",
                     disable_feedback=True)

    await msg.send()
    vector_db = await cl.make_async(get_vector_db)(file)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    def generate_answer_from_retrieval(query: str, vector_db: Chroma) -> str:
        # Truy vấn dữ liệu từ hệ thống retrieval
        embedding_query = model.encode(query)['dense_vecs']
        results = vector_db.query(
            query_embeddings=embedding_query,
            n_results=3
        )
        # Kết hợp dữ liệu truy xuất với câu hỏi và tạo prompt
        prompt = "Given the following table, answer the question:\n\n"
        for doc in results['doc']:
            prompt += f"{doc['content']}\n\n"

        prompt += f"Question: {query}\nAnswer:"

        # Sinh câu trả lời từ GPT-2
        answer = generate_answer(prompt)
        return answer

    chain = ConversationalRetrievalChain.from_llm(
        llm=generate_answer_from_retrieval,  # Sử dụng hàm trả lời từ retrieval
        chain_type="stuff",  # Chọn loại chain
        retriever=vector_db.,
        memory=message_history,
        return_source_documents=True  # Trả lại tài liệu nguồn (nếu cần)
    )

    msg.content = f"'{file.name}' processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["sources_documents"]
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content,
                        name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()
