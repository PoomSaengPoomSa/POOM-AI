import os

def main():
    log_path = r"C:\Users\subeen\.gemini\antigravity-ide\brain\e0261126-b0aa-42dd-a9fc-f9f48b30a849\.system_generated\tasks\task-2283.log"
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            print(f.read())
    else:
        print("Log file not found.")

if __name__ == '__main__':
    main()
