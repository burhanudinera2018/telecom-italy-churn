# inspect_format.py
print("=" * 60)
print("🔍 INSPECTING RAW TELECOM ITALIA FILE FORMAT")
print("=" * 60)

# Baca 10 baris pertama dari file
file_path = "data/telecom-italia-sample/sms-call-internet-mi-2013-11-01.txt"

print(f"\n📄 Reading first 10 lines of: {file_path}")
print("=" * 60)

with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        print(f"Line {i+1}: {repr(line[:200])}")  # Tampilkan 200 karakter pertama

print("\n" + "=" * 60)