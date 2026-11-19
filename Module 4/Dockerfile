FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy dependency files
#COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
#alternative since I dnt have python 3.10 in my system, else would have used pipfile directly
RUN pip install pandas numpy scikit-learn pyarrow

# Copy prediction script
COPY ["starter.py", "./"]

# Run the script
ENTRYPOINT ["python", "starter.py"]