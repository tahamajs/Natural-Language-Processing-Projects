**In the Name of God **^1^

**University of Tehran **^2^Faculty of Electrical and Computer Engineering ^3^Natural Language Processing Course ^4^Homework 6 ^5^

**+3**

- **Course Instructor:** Dr. Heshaam Faili ^6^
- **Head Teaching Assistants:** Milad Mohammadi and Amirhossein Safdarian ^7^
- **Assignment Designers:** Mehdi Sabour and Hossein Lashgari ^8^
- **Date:** Dey 1404 (January 2026) ^9^

---

**Table of Contents **^10^

_(Note: The table of contents lists sections that are detailed below)._

---

**Introduction **^11^

**In recent years, \*\***Large Language Models (LLMs)** have evolved from simple chatbots into powerful reasoning engines capable of executing complex **Workflows**^12^. **While early chatbots like ChatGPT could generate code snippets upon request, they lacked \***\*Agency** to interact with the real world^13^. **They could not search through system files, run a compiler to check for errors, or execute a \*\***Test Suite** to verify their logic^14^. **This limitation meant that the human user always had to act as the \*\* **Integrator** **—copying code, running it, returning errors to the model, and manually repeating this loop**^15^.

**In this assignment, you will design and implement a \*\***Coding Agent**^16^. **Specifically, this agent is capable of searching within a \*\* **Codebase** **, debugging errors, adding new features, and executing tests and code**^17^. **In essence, this agent will have access to a set of tools (developed by you) to perform the aforementioned capabilities**^18^. **To implement the agent and the required tools, you will use the \*\***LangGraph** and **LangChain** frameworks^19^. **You will also use Large Language Models via API as the core language processor without the need for fine-tuning**^20^. **Our proposed model is \*\* **gpt-5-mini** **, but you may use other models as well**^21^.

**+1**

**The method of communication between the user and the agent will be through a \*\***Command Line Interface (CLI)**^22^. **The code for a ready-made sample CLI is available in the \*\*`<span class="citation-617">coding-agent/src</span>` folder, which you can use and expand upon, or you can develop your own custom CLI from scratch using other tools^23^.

**+1**

**Assignment Objectives:**

- **Working with the \*\***LangGraph** framework and designing **Stateful\*\* graphs^24^.
- **Working with Command Line creation tools and session management**^25^.
- **Designing appropriate tools to equip the agent with capabilities for reading, writing, searching files, and executing commands**^26^.
- **Implementing \*\***Human In The Loop\*\* to prevent file changes and command executions without user permission^27^.
- **Evaluating the agent on software projects that contain problems and bugs**^28^.

---

**CodingAgent (100 Points) **^29^

**As mentioned in the introduction, the \*\***Coding Agent** acts as a programming assistant to help the user fix problems and create new features^30^. **By specifying the project path on the system for the agent and interacting with it via the CLI using natural language requests, the agent will begin examining files, reading them, and executing/testing them using the tools at its disposal to satisfy the user's request by modifying the code\*\*^31^.

**The components of the agent that must be implemented are described below**^32^:

#### 1.

Tool Development (25 Points) ^33^

**For the agent to interact with its environment (i.e., the user's project), functions (tools) need to be implemented using \*\***LangGraph**^34^. **The set of required tools includes the following (you are not limited to these and may implement other tools if needed)\*\*^35^:

**+1**

- **Read File:** This tool allows the agent to read the contents of a specific text file^36^.
- **Create File:** This tool allows the agent to create a new file^37^.
- **Overwrite File:** This tool allows the agent to overwrite an existing file^38^.
- **List Files:** This tool allows the agent to view the contents (files and folders) of a specific directory^39^. **The existence of this tool is essential for the agent to understand the project structure**^40^.
  **+1**
- **Search File:** Returns the paths of existing files with a specified name to the agent^41^.
- **Execute Shell Commands:** This tool allows the agent to execute necessary code and commands^42^. **The result or execution error is returned to the agent**^43^.
  **+1**

#### 2.

Graph Design and State Management (25 Points) ^44^

**The agent's state graph plays the most important role in how information is processed and how tools are used efficiently**^45^. **The more appropriate this graph is, the agent will reach the correct answer consuming fewer input and output tokens**^46^. **Therefore, using an appropriate architecture plays a decisive role in the agent's performance**^47^. **You must implement the nodes and edges of the graph using \*\***LangGraph** and by selecting an appropriate architecture such as **ReAct\*\* **, ** **Plan and Execute** **, or \*\***Reflexion\*\*^48^.

**+3**

#### 3.

Human In The Loop Development (15 Points) ^49^

An automated agent that has write access and command execution capabilities can be dangerous50. To prevent unwanted changes or the execution of malicious commands, you must implement a human supervisory mechanism51.

To do this, first identify sensitive tools52. When the agent requests to use these tools, the graph execution must stop, and by showing the tool name and its Arguments to the user in the CLI, it must ask for user confirmation (Yes/No)53.+2

- **If confirmed, the tool is executed, and its output is returned to the agent**^54^.
- **If not confirmed, without executing the tool, an error message must be returned to the agent as the tool output so the agent realizes it is not allowed to do that work and finds another way to solve the problem**^55^.

#### 4.

Memory Management (10 Points) ^56^

**One of the essentials in chatting with an agent is creating memory for it**^57^. **The agent must be able to see previous messages in a conversation by including them in the Context to use their information if needed and maintain the continuity of responses**^58^. **You can use \*\***LangGraph\*\* to implement this^59^.

#### 5.

Development and Connection to CLI (15 Points) ^60^

**You must use a \*\***Command Line Interface** for the agent to communicate with the user^61^. **After designing the graph and tools, you must connect them appropriately to the CLI**^62^. **A ready-made sample of the CLI is available in the **`<span class="citation-577">coding-agent/src</span>` folder, which is explained in detail in the next section^63^. **You can use and expand it if needed or design it from scratch**^64^. **Note that all user interaction with the agent takes place through the CLI, so display the agent's response and intermediate status (like tool execution) to the user in an appropriate format\*\*^65^.

#### 6.

Agent Evaluation (10 Points) ^66^

**Three small software projects have been designed for evaluating the agent's performance, located in the **`<span class="citation-573">coding-agent/test_projects</span>` folder^67^. **These projects contain bugs that cause them to malfunction**^68^. **You must run the agent on each of these projects separately and ask it in natural language to find and fix the bugs**^69^. **A **`<span class="citation-570">test</span>` file is designed for each project that helps the agent continue fixing problems until the tests pass^70^.

**+2**

#### 7.

Bonus Items (15 Points) ^71^

- **Usage Tracker (5 Points):** Implement and add a system to the agent to calculate the input and output token consumption in each turn and the total conversation, displaying it to the user^72^.
- **Smart Context Management (5 Points):** One problem with coding agents is the Context Window filling up with the content of long files and errors^73^. **Design a mechanism that, after finishing work with a file or detecting that the file content is no longer needed, removes it from the agent's memory or summarizes it to save costs**^74^.
- **Session Save and Restore (5 Points):** Add a feature to the agent and CLI that allows the user to save the current state of the conversation (e.g., in a JSON file) and resume work from that point later^75^. **This feature must restore both the message history and the internal graph state**^76^.

---

**Setup and Used Libraries **^77^

**Project Setup **^78^

**To set up the project located in the **`<span class="citation-561">coding-agent</span>` folder, preferably use **Python 3.11** or higher^79^.

1. **First, create a \*\***Virtual Environment\*\*^80^.
2. **Then, to install the project and dependent libraries, use the following command: **`<span class="citation-559">pip install -e .</span>`^81^.

   - **This command reads the **`<span class="citation-558">pyproject.toml</span>` file and installs the `<span class="citation-558">coding-agent</span>` package on the virtual environment^82^.

3. **After installation, you can run the agent with the command **`<span class="citation-557">coding-agent</span>`^83^. **If installed correctly, you will see the following output**^84^:
   **Plaintext**

   ```
   Usage: coding-agent [OPTIONS] COMMAND [ARGS]...
   Professional CLI-based coding agent with LangGraph.
   Run 'coding-agent chat' to start an interactive session.
   ...
   ```

4. To start a conversation with the agent, use the chat command8585. You must specify the target project path (the project the agent is supposed to work on) using -p86868686:
   coding-agent chat -p ./my-project87.+2

After executing this command, you enter the Interactive CLI environment designed with the Rich library88. Here you can make requests to the agent in natural language89.

By studying the codes in src/coding_agent/cli, you can expand this environment and add other features to the user interface90.

To use the API Key, preferably define it in a .env file similar to env.example and use it as an environment variable91.+3

**Used Libraries **^92^

- **LangGraph Framework:** This powerful library is designed for building intelligent, multi-step applications with LLMs^93^. **Unlike simple linear chains, LangGraph allows you to design your agent logic as a graph that includes loops**^94^. **This feature is vital for implementing architectures requiring "thought, action, and review" (like when an agent must retry after an error)**^95^. **In this assignment, you will use LangGraph to define decision nodes, tool nodes, and agent memory management**^96^.
  **+2**
- **Rich Library and UI Components:** The Rich library is a powerful Python tool responsible for converting old black-and-white terminals into interactive, colored, and structured environments^97^.
  - **Console:** Replaces the standard Python `<span class="citation-542">print</span>` object^98^. **Used to print formatted text using tags like **`<span class="citation-541">[bold red]</span>` to highlight errors^99^.
    **+1**
  - **Panel:** Used to frame text and create distinct visual blocks^100^. **Messages, help, and agent responses are placed inside a Panel**^101^.
    **+1**
  - **Prompt:** An advanced tool for receiving user input^102^. **It supports default text display and input validation**^103^.
    **+1**
  - **Table:** Used to display structured data in an organized and readable table format^104^.

---

**Evaluation Projects **^105^

**To evaluate your intelligent agent, three defective software projects have been designed**^106^. **Each includes source files, data, and a test suite that currently fails**^107^. **The agent's task is to enter the project folder like a software engineer, understand its structure, run tests, and fix existing bugs by modifying the code**^108^.

**+2**

#### 1.

Legacy CSV Processor ^109^

**This is an old command-line tool tasked with processing CSV files containing financial transactions**^110^. **The program must read the file, filter successful transactions, and then calculate total revenue and average transactions**^111^. **The agent must check the project code, run tests, and fix bugs related to ** **data types** **, ** **loading methods** **, and \*\***edge cases\*\*^112^.

- **Important Note:** This code is **not allowed** to use the `<span class="citation-527">pandas</span>` library, and file parsing must be done manually by `<span class="citation-527">file_handler.py</span>`^113^.
- **Files:**
  - `<span class="citation-526">src/file_handler.py</span>`: Reads text files, separates commas, builds list of dictionaries^114^.
  - `<span class="citation-525">src/analyzer.py</span>`: Handles mathematical calculations^115^.
  - `<span class="citation-524">main.py</span>`: Entry point^116^.
  - `<span class="citation-523">data/transactions.csv</span>`: Standard file with `<span class="citation-523">id</span>`, `<span class="citation-523">amount</span>`, `<span class="citation-523">status</span>`, `<span class="citation-523">date</span>`^117^.
  - `<span class="citation-522">data/empty.csv</span>`: Empty file to test stability^118^.
- **Tests (`tests/test_analyzer.py`):**
  - `<span class="citation-521">test_load_transactions</span>`: Checks data loading correctness^119^.
  - `<span class="citation-520">test_calculate_total_revenue</span>`: Checks if total revenue equals exactly **1350.50**^120^.
  - `<span class="citation-519">test_handle_empty_file</span>`: Checks that the program doesn't crash on empty files and returns `<span class="citation-519">0.0</span>`^121^.
  - `<span class="citation-518">test_get_average_transaction</span>`: Checks average calculation correctness^122^.

#### 2.

TinyRAG Search System ^123^

**A minimal implementation of a \*\***RAG** system that chunks text documents and searches using word frequency vector similarity^124^. **However, search results are very poor quality**^125^. **The agent must check the mathematical logic and text processing to correct algorithmic errors\*\*^126^.

**+1**

- **Files:**
  - `<span class="citation-513">ingest/loader.py</span>`: Loads text files^127^.
  - `<span class="citation-512">ingest/chunker.py</span>`: Splits text based on words^128^.
  - `<span class="citation-511">retrieval/embedding.py</span>`: Converts text to vectors (word count)^129^.
  - `<span class="citation-510">retrieval/search.py</span>`: Calculates similarity between Query and documents^130^.
  - `<span class="citation-509">database/store.py</span>`: In-memory database^131^.
  - `<span class="citation-508">data/knowledge_base.txt</span>`: Test file with meaningful sentences and noise^132^.
- **Tests (`tests/test_tinyrag.py`):**
  - `<span class="citation-507">test_chunking_logic</span>`: Checks chunking performance^133^.
  - `<span class="citation-506">test_search_quality</span>`: Checks if searching a keyword finds the relevant document instead of noise^134^.
  - `<span class="citation-505">test_similarity_score</span>`: Checks that a document's similarity with itself is exactly **1.0**^135^.

#### 3.

Bigram Language Model ^136^

**A statistical \*\***Bigram** language model that predicts the next word based solely on the previous one^137^. **Used for text generation and sentence scoring**^138^. **Currently, it fails on new words, loses precision in long sentences, and generates repetitive text**^139^. **The agent must correct the probability math and improve the generation mechanism\*\*^140^.

**+1**

- **Files:**
  - `<span class="citation-499">model/bigram.py</span>`: Main class for training and probability calculation^141^.
  - `<span class="citation-498">model/tokenizer.py</span>`: Helper to tokenize text^142^.
  - `<span class="citation-497">generation/sampler.py</span>`: Generates text word-by-word^143^.
  - `<span class="citation-496">data/corpus.txt</span>`: Simple sentences for training^144^.
- **Tests (`tests/test_bigram.py`):**
  - `<span class="citation-495">test_smoothing_unseen_bigrams</span>`: Checks that unseen word pairs do not have zero probability^145^^145^^145^^145^.
    **+1**
  - `<span class="citation-494">test_underflow_long_sequence</span>`: Checks if long sentence probability tends to zero (underflow)^146^^146^^146^^146^.
    **+1**
  - `<span class="citation-493">test_generation_diversity</span>`: Checks that 10 generation attempts produce different outputs^147^.

---

**+4**

**Submission Notes (Must Read) **^148^

1. **Saving Logs:** After full implementation, during agent execution on evaluation projects, the conversations and intermediate states (tools and state) must be saved in a file (e.g., json, txt)^149^.
2. **Presentation Video:** Prepare a **5 to 8-minute video** at the end showing the general implementation (graph architecture, tools) and the agent's performance on evaluation projects^150^. **No need for minute details, but show main capabilities**^151^.
   **+1**
3. **Executability:** Your code must be executable. **If using libraries other than those mentioned, state them and any specific settings required**^152^.
4. **Zip File:** All results (code, corrected evaluation projects, video, logs) must be submitted in a zip file named `<span class="citation-487">NLP-HW6-StudentID</span>`^153^.
5. **Late Submission:** There is **NO** late submission. **The final assignment does not include a grace period**^154^.
6. **Individual Work:** This is an individual assignment. **If similarity is found, all participants get a zero and are reported to the professor**^155^.
7. **Contact:** If issues arise, contact:
   - `<span class="citation-484">lashgari.hn@gmail.com</span>`^156^
   - `<span class="citation-483">msabour.official@gmail.com</span>`^157^

**Dates:**

- **Assignment Upload:** 14 Dey 1404 (Jan 4, 2026)^158^.
- **Deadline (No Penalty):** 16 Bahman 1404 (Feb 5, 2026)^159^.
- **Late Deadline:** None!^160^.
