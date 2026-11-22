FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi "uvicorn[standard]"
COPY data.py model.py app.py ./
COPY static ./static
COPY outputs/model.pt ./outputs/model.pt
ENV MODEL_PATH=outputs/model.pt
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
