from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = "The solar system is a vast, gravitationally bound planetary system that formed roughly 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. At its absolute epicenter sits the Sun, a yellow dwarf star that contains more than 99.8% of the entire system's mass and exerts the immense gravitational pull required to keep all neighboring celestial bodies in orbit. Circling the Sun are eight major planets, which astronomers divide into two distinct categories: the four rocky, dense terrestrial planets of the inner solar system—Mercury, Venus, Earth, and Mars—and the four massive jovian planets of the outer solar system, further split into the gas giants Jupiter and Saturn, and the ice giants Uranus and Neptune. Beyond these major worlds, the solar system is populated by numerous dwarf planets like Pluto and Ceres, hundreds of planetary moons, a dense asteroid belt located between Mars and Jupiter, and trillions of icy bodies within the distant Kuiper Belt and Oort Cloud, all moving together through the Orion Arm of the Milky Way Galaxy."

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)
result = splitter.split_documents(docs)
# result = splitter.split_text(text)

print(result[1].page_content)
print(result[0])