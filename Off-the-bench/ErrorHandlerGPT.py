from chatgpt_wrapper import ChatGPT


class ErrorHandlerGPT:

    def __init__(self, api_key):
        self.api_key = api_key
        self.bot = ChatGPT()

    def ask_chatgpt(self, question):
        success, response, message = self.bot.ask(question)
        return response


# %%

handler = ErrorHandlerGPT(api_key="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")

try:
    # Example of code that can produce an error
    my_list = [1, 2, 3]
    print(my_list[3])  # IndexError: list index out of range
except Exception as e:
    error_message = str(e)
    print(f"Error encountered: {error_message}")

    question = f"I encountered the following error: '{error_message}'. How can I fix it?"
    answer = handler.ask_chatgpt(question)
    print(f"Suggested solution: {answer}")
