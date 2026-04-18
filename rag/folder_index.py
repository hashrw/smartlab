from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


def build_bibliography_index(folder_path: str):
    documents = SimpleDirectoryReader(
        input_dir=folder_path,
        recursive=True,
        filename_as_id=True,
        required_exts=[".pdf", ".txt", ".md"],
        file_metadata=lambda f: {
            "source": "bibliography",
            "file_name": Path(f).name,
        },
    ).load_data()

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=120)
    nodes = splitter.get_nodes_from_documents(documents)

    return VectorStoreIndex(nodes)