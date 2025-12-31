# ১. পাইথন লাইটওয়েট ইমেজ ব্যবহার করা হচ্ছে
FROM python:3.9-slim

# ২. সিস্টেম আপডেট এবং FFmpeg ইনস্টল করা (ভিডিও মার্জ করার জন্য জরুরি)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ৩. অ্যাপের জন্য ডিরেক্টরি তৈরি
WORKDIR /app

# ৪. রিকয়ারমেন্টস ফাইল কপি এবং লাইব্রেরি ইনস্টল
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ৫. প্রোজেক্টের সব ফাইল কপি করা
COPY . .

# ৬. কুকিজ ফাইল নিশ্চিত করা (আপনার এরর ফিক্স করার জন্য)
# যদি ফাইলের নাম cookies.txt হয় তবে সেটি কপি হবে
COPY cookies.txt* ./ 

# ৭. স্ট্যাটিক ফাইল ম্যানেজমেন্ট
# index.html ফাইলটি static ফোল্ডারে না থাকলে এটি স্বয়ংক্রিয়ভাবে সরিয়ে নেবে
RUN mkdir -p static && (mv index.html static/ 2>/dev/null || true)

# ৮. পোর্ট কনফিগারেশন (Render স্বয়ংক্রিয়ভাবে PORT এনভায়রনমেন্ট ভেরিয়েবল দেয়)
ENV PORT=5000
EXPOSE 5000

# ৯. Gunicorn প্রোডাকশন সার্ভার দিয়ে অ্যাপ চালু করা
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} server:app"]
