# from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# text = """
# # Project Name: Smart Student Tracker

# A simple Python-based project to manage and track student data, including their grades, age, and academic status.


# ## Features

# - Add new students with relevant info
# - View student details
# - Check if a student is passing
# - Easily extendable class-based design


# ## Tech Stack

# - Python 3.10+
# - No external dependencies


# ## Getting Started

# 1. Clone the repo
#    git clone https://github.com/your-username/student-tracker.git
# """

# splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.MARKDOWN,
#     chunk_size=200,
#     chunk_overlap=0,
# )

# chunks = splitter.split_text(text)

# print(len(chunks))
# print(chunks[0])


from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0


student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])