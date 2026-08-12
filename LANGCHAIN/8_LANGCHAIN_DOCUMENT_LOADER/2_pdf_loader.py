from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))              # number of pages in the PDF
print(docs[0].page_content)   # text content of page 1
print(docs[1].metadata)       # {'source': 'dl-curriculum.pdf', 'page': 1}