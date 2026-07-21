import subprocess

scripts = [
    "utils/github_collector.py",
    "utils/stackoverflow_collector.py",
    "utils/python_discussions.py",
    "utils/pytorch_forums.py",
    "utils/numpy_collector.py",
    "utils/fastapi_collector.py",
]

for script in scripts:

    print("=" * 60)
    print("Running:", script)

    subprocess.run(["python", script])

print("\nAll collectors finished.")