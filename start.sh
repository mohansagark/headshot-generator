#!/bin/bash

# Create startup script for cloud deployment
mkdir -p ~/.streamlit/

# Create Streamlit config
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
port = \$PORT

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
EOF

# Run the Streamlit app
streamlit run app.py
