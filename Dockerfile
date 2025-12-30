# পাইথন বেস ইমেজ ব্যবহার করা হচ্ছে
FROM python:3.9-slim

# সিস্টেমে FFmpeg ইনস্টল করা
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# কাজের ডিরেক্টরি সেট করা
WORKDIR /app

# রিকয়ারমেন্টস কপি এবং ইনস্টল করা
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# সব ফাইল কপি করা
COPY . .

# 'static' ফোল্ডার তৈরি করা এবং index.html সেখানে সরানো (যদি না থাকে)
RUN mkdir -p static && mv index.html static/ 2>/dev/null || true

# Render এর পোর্ট কনফিগারেশন
ENV PORT=5000
EXPOSE 5000

# Gunicorn দিয়ে অ্যাপ রান করা
CMD gunicorn --bind 0.0.0.0:$PORT server:app
