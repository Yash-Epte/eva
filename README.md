
---

# EVA

EVA is a comprehensive Streamlit application designed to handle CSV operations, manage home functionalities, and process PDF files. This repository contains Python scripts that collectively deliver these functionalities through an intuitive web interface.
![image](https://github.com/Yash-Epte/eva/assets/121223452/300ec7e1-6e0c-44c0-a33d-48bb5dc0fd28)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Chat With PDF](#chat-with-pdf)
  - [Chat With CSV](#chat-with-csv)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

EVA is a versatile Streamlit application aimed at providing efficient solutions for CSV management, home automation processes, and PDF handling. The project is structured to ensure easy navigation and usage of the different scripts.

## Features

- **Chat With PDF**: Upload PDFs and generate text as well as questions based on the content using OpenAI LLM Model.
- **Chat With CSV**: Upload university mark sheet CSV files, visualize the data, and generate text and questions based on the content using OpenAI LLM Model and MongoDB for vector database functionalities.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

Ensure you have the following packages installed:
- streamlit
- pandas
- PyPDF2
- openai
- pymongo
- Any other dependencies listed in the scripts

## Usage

To run the EVA Streamlit application, execute the following command in your terminal:

```bash
streamlit run Home.py
```

### Chat With PDF

1. **Upload a PDF file**: Drag and drop your PDF file into the designated area.
2. **Process the PDF**: The application extracts text from the PDF and generates questions based on the content.
3. **Interact**: Use the chat interface to ask questions and receive answers from the processed PDF content.

![image](https://github.com/Yash-Epte/eva/assets/121223452/84230ad7-1c5d-4dda-8f84-a9825ee76cdf)



### Chat With CSV

1. **Upload a CSV file**: Drag and drop your university mark sheet CSV file into the designated area.
2. **Visualize and Analyze**: The application visualizes the CSV data and generates questions based on the content. It uses MongoDB as a vector database for efficient data handling and querying.
3. **Interact**: Use the chat interface to ask questions and receive answers from the processed CSV data.

<img width="866" alt="image" src="https://github.com/Yash-Epte/eva/assets/121223452/3e506f77-539d-4ba6-a91b-a9654a3903e3">


## Contributing

We welcome contributions to enhance EVA. To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback, please contact [epteyash777@gmail.com](mailto:your-email@example.com).

---

