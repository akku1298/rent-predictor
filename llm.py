import os
import time

from google import genai


def get_client(timeout: int = 30):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Streamlit Cloud secrets also land in env via st.secrets; handled in app.py
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def chat(prompt, model="gemini-3.8-flash", system="You are a helpful Indian rental advisor.", retries=3):
    """Reusable LLM call with timeout, retries, rate-limit backoff."""
    full_prompt = f"{system}\n\nUser: {prompt}" if system else prompt
    last_err = None
    for i in range(retries):
        try:
            client = get_client()
            r = client.models.generate_content(model=model, contents=full_prompt)
            return r.text
        except Exception as e:
            last_err = e
            msg = str(e)
            # 429 / quota / overloaded -> backoff, else retry once then raise
            if any(x in msg for x in ("429", "quota", "overload", "503", "rate")) and i < retries - 1:
                time.sleep(2**i)
                continue
            if i < retries - 1:
                time.sleep(1)
                continue
            raise
    raise last_err
