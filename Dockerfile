# ১. পাইথন লাইটওয়েট ইমেজ ব্যবহার
FROM python:3.9-slim

# ২. সিস্টেম এনভায়রনমেন্ট সেটআপ
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# ৩. সিস্টেম আপডেট এবং প্রয়োজনীয় ডিপেন্ডেন্সি ইনস্টল
# এখানে nodejs এবং npm যুক্ত করা হয়েছে yt-dlp এর JavaScript runtime এরর ফিক্স করতে
RUN apt-get update && apt-get install -y \
    ffmpeg \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ৪. অ্যাপ ডিরেক্টরি তৈরি
WORKDIR /app

# ৫. রিকয়ারমেন্টস ফাইল কপি এবং লাইব্রেরি ইনস্টল
# yt-dlp সব সময় লেটেস্ট রাখতে বিল্ডের সময় আপডেট করা হচ্ছে
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade yt-dlp

# ৬. প্রোজেক্টের সব ফাইল কপি করা
COPY . .

# ৭. কুকিজ ফাইল এবং স্ট্যাটিক ফোল্ডার নিশ্চিত করা
# cookies.txt ফাইলটি থাকলে সেটি কপি হবে, না থাকলে এরর দেবে না
RUN mkdir -p static temp_downloads && \
    (mv index.html static/ 2>/dev/null || true) && \
    touch cookies.txt

# ৮. পারমিশন সেট করা (Render-এ ফাইল রাইট করার সুবিধার জন্য)
RUN chmod -R 777 /app/temp_downloads

# ৯. পোর্ট এক্সপোজ করা
EXPOSE 5000

# ১০. Gunicorn দিয়ে অ্যাপ রান করা
# Gunicorn ব্যবহার করা হয়েছে যাতে মাল্টিপল রিকোয়েস্ট হ্যান্ডেল করা যায়
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120 server:app"]
