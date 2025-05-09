!pip install streamlit folium streamlit-folium
!npm install -g localtunnel
from google.colab import files
uploaded = files.upload()

!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
