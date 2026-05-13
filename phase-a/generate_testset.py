import os
import asyncio
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def generate():
    print("🚀 Starting Test Set Generation (Refined Distribution & Vietnamese)...")
    
    # 1. Load specific documents
    documents = []
    target_files = [
        "data/lab24-student-edition.md",
        "data/tailieutaphuanAI_THPT.md"
    ]
    
    for file_path in target_files:
        if os.path.exists(file_path):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
            print(f"✅ Loaded: {file_path}")
        else:
            print(f"⚠️ Warning: File not found: {file_path}")

    if not documents:
        print("❌ Error: No documents loaded.")
        return

    # 2. Setup wrappers
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # 3. Initialize Generator
    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings
    )

    # 4. Define Custom Distribution (50/25/25) & Enforce Vietnamese
    # Thêm instructions tiếng Việt vào llm_context
    vn_context = "Hãy luôn tạo câu hỏi và câu trả lời bằng tiếng Việt chính xác, tự nhiên."
    
    query_dist = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, llm_context=vn_context), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm, llm_context=vn_context), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm, llm_context=vn_context), 0.25)
    ]

    # 5. Generate test set
    print("⏳ Generating 50 questions (this may take 10-15 minutes)...")
    
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=50,
        query_distribution=query_dist
    )

    # 6. Save to CSV
    output_path = "phase-a/testset_v1.csv"
    os.makedirs("phase-a", exist_ok=True)
    testset.to_pandas().to_csv(output_path, index=False)
    print(f"✨ Done! Test set saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(generate())
