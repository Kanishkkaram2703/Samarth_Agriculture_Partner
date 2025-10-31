# 🌾 Project Samarth – AI Q&A Platform for Agriculture & Climate Insights

## Objective
To create an intelligent Q&A platform that helps users query agricultural and climate data using  local datasets.

## Problem Statement
Farmers and policymakers often lack integrated access to data-driven insights on rainfall, crop yields, and agricultural trends. Project Samarth bridges this gap using AI and open data.

## Features
- Agriculture & rainfall dataset integration
- Smart search with auto-source citation
- Modern web interface (Flask + HTML + JS)
- Environment variable protection (.env)
- Realtime fetching of Indian agriculture insights

## Project Structure

samarth_qna/
│
├── app.py   # Flask backend
├── templates/index.html   # Frontend
├── static/style.css   # Styling
├── datasets/   # CSV data
└── requirements.txt   # Dependencies

## 🚀 Setup Instructions
1. Clone or extract the project to `D:\Internship_Project\samarth_qna`
2. Run `pip install -r requirements.txt`
3. Create `.env` file with:
4. Run `python app.py`
5. Open in browser: `http://127.0.0.1:5000`

## Future Enhancements
- Add farmer chat assistant (voice enabled)
- Integrate real IMD APIs
- Add data visualization dashboards
- Deploy on Render or Streamlit Cloud

Made by Kanishk 🤍🤍