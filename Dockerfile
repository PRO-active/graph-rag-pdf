# ベースイメージを指定
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードをコピー
COPY . .

# コンテナ起動時に実行するコマンドを指定
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
