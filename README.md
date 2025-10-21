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
---

### Extraction Pipeline

The main fundamental technique of my apporach in doing extraction of the financial statement pdf is by using the each
coordinates of words and the rectangle the library can detect. Library that I used on this particular approach is *
*pdfplumber**. The package has a built in extract table function which could be used to parse non-complicated table,
however it is not the case for the financial statement pdf, since there a lot of different format and nested column
values that should be parsed like the table in the code of account [1210000]. Below are the step by step explanation of
the extraction pipeline.

1. The start of the extraction is to open the PDF by using pdfplumber and detect the start of code of account until the
   end, after determining each sections for each code of account, the process of extraction will start.
2. Word from every page is extracted and paired to other words that can be assume as one sentence in that line. To
   determine if the words in one line is a sentence or not determined by the size of the word and the gap each word
   have, the logic is specified in the function **create_clusters** in extraction pipeline
3. After getting every line which consists of words, the lines the next step is splitting this into sections depending
   on each table. The financial statement pdf each code of account can have multiple function that can be splitted
   horizontally or vertically, hence the details of the method can be seen in the function **split_section_per_table**
   and **split_section_per_table**
4. Result from the 3 steps above is a list of table (list of line), after successfully doing this the pipepline will
   continue by parsing each table, below is the step by step for parsing a table:
    1. Determine whether the function has nested column description like tables in code of account (10000000, 1210000,
       etc.)
    2. After determining the table type they each will undergo the appropriate function.
    3. Finding the coordinates for each column and row is done by using the **page.rects** attribute, a bigger height
       then the rectangle is a column barrier, and a smaller one is mapped to row. This is how the process determine
       values in a
       each row and column
    4. After mapping each line to each column and row the final step is processing it into a dict value, which in the
       function **nested_format_final_result** and **format_final_result**, the function for format nested result is
       determined by the indentation of the first column, if there is an indentation the process create a new dict
       inside the
       current dict. For the other format result function it is much more straight forward just the mapping of each
       value into each column description.

By visual examining, my approach will work 100% of the time if the format of rules that have been ste by each that I
created is not broken and validate if my extraction is correct, I did it by comparing the result from each code of
account extraction to the actual PDF. However, to automate this process I would use the equation of **Assets =
Liabilities + Equity**, this would take the categorization result to run for all extracted result.
---

### Categorization Pipeline

To categorize each data in all code of account I employs a prompting technique and the use of LLM. For context hardware
that I'm currently using for this take home test is not that powerful, hence there is no result for a fully labeled
data but there is an example of it in the folder labeled_data. The propmting technique can be seen in the file
categorize_prompt, and below is the step of our technique:

1. Format the description and title that is going to be categorize
2. Make an analysis of the description and title using LLM
3. Use the analysis from before to categorize the general label
4. Use the result from the previous 2 step and categorize the detailed label

For details in the prompting I use can be seen in the file, below are the labels that I used for the categorization.

```
{
    "Assets Current": ["Cash", "Inventory", "Receivable", "Prepaid", "Short-term"],
    "Assets Non Current": ["Property", "Equipment", "Intangible", "Long-term", "Investment"],
    "Liabilities Current": ["Payable", "Accrued", "Short-term loan", "Liability"],
    "Liabilities Non Current": ["Long-term", "Bond", "Lease liability", "Liability"],
    "Equity": ["Capital", "Retained earnings", "Reserves", "Share", "Dividend"],
    "Income Statement": ["Revenue", "Income", "Expense", "Cost", "Profit", "Loss"],
    "Others": []
}

```

---

### Limitation and Improvements

Since the baremetal that I am currently using, I am unable to categorize every data and in result unable to do
validation pipeline since the categorization is important for validating extracted data using the equation mentioned
above.

Improvements that I would to make the extraction 100 percent correct is to use the HTML files in the folder *
*data/inlineXBRL** which was obtained from IDX website also. HTML has a more structured way of formatting elements
making it more reliable to parse each data from all code of accounts

---
**created by William Christian**