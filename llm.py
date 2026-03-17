from config import LOCAL_LLM_URL, GOOGLE_API_KEY


def get_llm():
    if LOCAL_LLM_URL:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=LOCAL_LLM_URL,
            api_key="not-needed",
            model="local-model",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-09-2025", google_api_key=GOOGLE_API_KEY)
