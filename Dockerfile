# 彩票AI预测系统 Docker 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY examples/ ./examples/

# 创建必要的目录
RUN mkdir -p data model predict logs

# 设置权限
RUN chmod +x scripts/*.py

# 暴露端口（如果需要web服务）
# EXPOSE 8000

# 默认命令
CMD ["python", "examples/quick_start.py"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"