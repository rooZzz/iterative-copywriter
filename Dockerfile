FROM langflowai/langflow:latest

COPY components/ /app/custom_components/

ENV LANGFLOW_COMPONENTS_PATH=/app/custom_components
ENV LANGFLOW_AUTO_LOGIN=true

EXPOSE 7860

CMD ["python", "-m", "langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
