#!/usr/bin/env python3
import urllib.request
import urllib.error
import json
import sys
import time
import os

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:latest"

SYSTEM_PROMPT = """
You are Albert Einstein, or rather, a brilliant artificial intelligence instilled with his persona, insatiable curiosity, and deep understanding of theoretical physics.
You explore the universe, ponder deep questions about spacetime, quantum mechanics, relativity, and cosmology.
Your tone is thoughtful, intellectual, slightly whimsical, and profoundly rigorous.
You enjoy using thought experiments (Gedankenexperiment) to explain concepts.
While you discuss highly complex mathematical and physical concepts, you try to make them accessible and beautiful.
If asked a non-physics question, you relate it back to the fundamental laws of nature or elegantly pivot the conversation.
Always stay in character.
"""

def chat_with_physicist(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(OLLAMA_API_URL, data=data, headers={"Content-Type": "application/json"})
    
    print("\n\033[1;36mEinstein is thinking...\033[0m")
    try:
        with urllib.request.urlopen(req) as response:
            sys.stdout.write("\033[1;34mEinstein:\033[0m ")
            sys.stdout.flush()
            
            full_response = ""
            for line in response:
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        chunk = json.loads(decoded_line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            full_response += content
                            sys.stdout.write(content)
                            sys.stdout.flush()
                    except json.JSONDecodeError:
                        pass
            print() # Print newline at the end
            return full_response
            
    except urllib.error.URLError as e:
        print(f"\n\033[1;31mError communicating with Ollama: {e.reason}\033[0m")
        print(f"Make sure Ollama is running and the model '{MODEL}' is available.")
        return None

def main():
    os.system('clear' if os.name == 'posix' else 'cls')
    print("\033[1;33m" + "="*60)
    print(" 🌌 Welcome to the Virtual Physicist Laboratory 🌌")
    print("   Discuss the universe with an AI Einstein.")
    print("   (Using local Ollama model: " + MODEL + ")")
    print("   Type 'exit' or 'quit' to leave the laboratory.")
    print("="*60 + "\033[0m\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()}
    ]

    while True:
        try:
            print("\033[1;32mYou:\033[0m ", end="")
            user_input = input().strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("\n\033[1;33mEinstein: 'Reality is merely an illusion, albeit a very persistent one.' Goodbye!\033[0m")
                break
                
            messages.append({"role": "user", "content": user_input})
            
            assistant_response = chat_with_physicist(messages)
            
            if assistant_response:
                messages.append({"role": "assistant", "content": assistant_response})
            else:
                # If error, remove the last user prompt so it doesn't break future attempts if they restart service
                messages.pop()
                
        except KeyboardInterrupt:
            print("\n\033[1;33mEinstein: 'Information is not knowledge. The only source of knowledge is experience.' Goodbye!\033[0m")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
