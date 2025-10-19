# Financial Document Processor

**Project Dependencies**, In the development of the project, Python version in used is version 3.13.5 and I personally
used conda to create the environment. All project requirements is already documented inside the file requirements.txt

**Scripts for running application**, in this project I used argument parsing, the reason is to help the development
process when developing each pipeline (detail in other section) since without using any arguments there is a need in
changing the code first in the **main.py**. Below are the script that can be used for running the application:

``python main.py --action api``

``python main.py --action extract``

``python main.py --action categorize --code 1000000``

``python main.py --action summarize --code 1000000``

**How to run the application?** There is two ways of running the application which is going to be explained below

1. Without Docker, Please create a new environment with the same python version used detailed in this documentation.
   Then the script mentioned above can be used for running the application (do not forget to install the required
   package first.)
2. Docker, I have given a create a Docker script using **Dockerfile, and docker-compose.yml**, that can be used for
   runnign the application, the default action this method of running is API, to change the method just modify the
   Dockerfile.

**_Please note that you have created the .env file correctly._**

---

### Project Structure

In this section, I will explain the structure of the project detailing what is the use of each folder and how it
supports each other. I will also detail the key file in each folder briefly.

**main.py**, this file is the entry point of the whole project

**api.py**, this file serve as the entry point of the API

**_helpers_**, this folder will include files that is populated by method for side logics which going to help the main
logic in other folder. The purpose of this folder is for a more readable code.

**_llm_**, files in this folder is just focused in the use of LLM, the LLM that can now be used are Gemini LLM and
Ollama LLM. I am personally using Ollama LLM for the most part because of the free api_key query limit per day for
Gemini LLM.

**_pipelines_**, Folder that contains all files that is for the main logic

_The reason of this project structure is to keep it simple but still readable since the time that is given to do the
test is limited_

### Extraction Pipeline

### Categorization Pipeline

### Summarization Pipeline

### Limitation and Improvements
