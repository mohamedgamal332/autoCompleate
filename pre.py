import tkinter as tk
from tkinter import scrolledtext
import threading
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import traceback # For detailed error printing

# --- Configuration ---
MODEL_NAME = 'gpt2'  # CRITICAL: Use 'gpt2' for speed, especially on CPU.
                     # 'gpt2-medium', 'gpt2-large', 'gpt2-xl' will be much slower.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICTION_DELAY_MS = 1000  # Milliseconds to wait after user stops typing

# --- Optimizations for Speed ---
MAX_INPUT_TOKENS_FOR_PREDICTION = 64 # Only use the last N tokens of user input for context
                                     # GPT-2's max context is 1024. Shorter is much faster.
                                     # Adjust this based on performance.
MAX_NEW_TOKENS_TO_GENERATE = 3   # Generate only a few tokens, we'll pick the first word from them.
                                 # Generating 1 token is fastest if you only need the very next one.

# --- Global Variables ---
model = None
tokenizer = None
prediction_timer = None
is_model_loading = True
status_label = None # Initialize to None
root = None # Initialize to None
text_area = None # Initialize to None
prediction_label = None # Initialize to None


# --- Model Loading ---
def load_model():
    global model, tokenizer, is_model_loading, status_label
    print(f"Attempting to load {MODEL_NAME} model and tokenizer...")
    if status_label and root: # Check if GUI elements exist
        root.after(0, lambda: status_label.config(text=f"Loading {MODEL_NAME}... This may take time."))
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        print(f"Model {MODEL_NAME} loaded successfully on {DEVICE}.")
        if status_label and root:
            root.after(0, lambda: status_label.config(text=f"Model loaded on {DEVICE}. Ready."))
    except Exception as e:
        print(f"FATAL: Error loading model: {e}")
        traceback.print_exc()
        if status_label and root:
            root.after(0, lambda: status_label.config(text="Model loading FAILED. Check console."))
    finally:
        is_model_loading = False


# --- Prediction Logic ---
def predict_next_word(text_so_far):
    if not model or not tokenizer or not text_so_far.strip():
        return ""

    print(f"\n[{time.strftime('%H:%M:%S')}] --- Starting Prediction ---")
    print(f"Input text (last 100 chars): '...{text_so_far[-100:]}'")
    overall_start_time = time.time()

    try:
        # 1. Tokenize the entire input to identify the part we want
        #    We do this to correctly select the last MAX_INPUT_TOKENS_FOR_PREDICTION tokens
        full_input_token_ids = tokenizer.encode(text_so_far) # Just get IDs

        # 2. Select the last N tokens for actual model input
        if len(full_input_token_ids) > MAX_INPUT_TOKENS_FOR_PREDICTION:
            start_index = len(full_input_token_ids) - MAX_INPUT_TOKENS_FOR_PREDICTION
            truncated_token_ids_list = full_input_token_ids[start_index:]
            print(f"Input context truncated from {len(full_input_token_ids)} to {len(truncated_token_ids_list)} tokens.")
        else:
            truncated_token_ids_list = full_input_token_ids
        
        # Convert list of IDs to tensor for the model
        input_ids_tensor = torch.tensor([truncated_token_ids_list]).to(DEVICE)

        # 3. Create attention mask for the truncated input
        #    For a single, unpadded sequence fed to generate, it's all 1s.
        attention_mask_tensor = torch.ones_like(input_ids_tensor).to(DEVICE)

        prep_time = time.time()
        print(f"Tokenization & Prep time: {prep_time - overall_start_time:.4f}s")

        # 4. Generate output
        with torch.no_grad():
            generation_start_time = time.time()
            outputs = model.generate(
                input_ids_tensor,
                attention_mask=attention_mask_tensor,
                max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE, # Generate only a few new tokens
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # For initial speed test, use greedy search (fastest)
                do_sample=False,
                # If greedy works, you can try enabling sampling:
                # do_sample=True,
                # top_k=50,
                # top_p=0.95,
                # temperature=0.7
            )
            generation_end_time = time.time()
            print(f"Model.generate() time: {generation_end_time - generation_start_time:.4f}s")

        # 5. Decode only the newly generated tokens
        #    outputs[0] contains input_ids_tensor + generated_ids
        #    input_ids_tensor.shape[-1] is the length of the input sequence fed to the model
        predicted_token_ids = outputs[0][input_ids_tensor.shape[-1]:]
        predicted_sequence = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)

        decode_time = time.time()
        print(f"Decoding time: {decode_time - generation_end_time:.4f}s")
        print(f"Raw predicted sequence from model: '{predicted_sequence}'")

        # 6. Extract the first word
        predicted_words = predicted_sequence.strip().split()
        if predicted_words:
            first_word = predicted_words[0]
            print(f"Returning first predicted word: '{first_word}'")
            return first_word
        
        print("No valid words predicted (e.g., only punctuation or empty).")
        return ""

    except Exception as e:
        print(f"ERROR during prediction: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return "[Error]"
    finally:
        overall_end_time = time.time()
        print(f"Total prediction function time: {overall_end_time - overall_start_time:.4f}s")
        print(f"--- Prediction Finished ---")


# --- GUI Event Handlers ---
def on_text_change(event=None):
    global prediction_timer, is_model_loading, status_label, root, prediction_label
    if is_model_loading:
        if status_label and root:
             root.after(0, lambda: status_label.config(text="Model is still loading, please wait..."))
        return

    if prediction_timer:
        if root: # Check if root exists
            root.after_cancel(prediction_timer)

    if prediction_label and root:
        root.after(0, lambda: prediction_label.config(text="Prediction: ... (thinking)"))

    if root: # Check if root exists
        prediction_timer = root.after(PREDICTION_DELAY_MS, perform_prediction)

def perform_prediction():
    global text_area, prediction_label, root
    if not text_area or not root: # Ensure GUI elements are available
        print("Error: text_area or root not initialized for perform_prediction")
        return

    current_text = text_area.get("1.0", tk.END).strip()
    if not current_text:
        if prediction_label:
            root.after(0, lambda: prediction_label.config(text="Prediction: (Type something)"))
        return

    # Run prediction in a separate thread to avoid freezing the GUI
    def prediction_task():
        print(f"\n[{time.strftime('%H:%M:%S')}] GUI triggered: Starting prediction_task in thread...")
        predicted_word = predict_next_word(current_text)
        print(f"[{time.strftime('%H:%M:%S')}] Thread finished: predict_next_word returned: '{predicted_word}'")

        def update_gui_with_prediction():
            if root and root.winfo_exists() and prediction_label: # Check if window and label still exist
                if predicted_word:
                    prediction_label.config(text=f"Prediction: {predicted_word}")
                else:
                    prediction_label.config(text="Prediction: -")
            else:
                print("GUI update skipped: Window or label no longer exists.")
        
        if root: # Ensure root exists before scheduling GUI update
            root.after(0, update_gui_with_prediction) # Update GUI from the main thread

    threading.Thread(target=prediction_task, daemon=True).start()


# --- GUI Setup ---
def setup_gui():
    global root, text_area, prediction_label, status_label # Make them accessible globally

    root = tk.Tk()
    root.title("GPT-2 Next Word Predictor")
    root.geometry("700x450") # Slightly wider for more text

    # Instructions
    instruction_label = tk.Label(root, text="Start typing. Prediction appears after a pause. Check console for logs.", pady=5)
    instruction_label.pack()

    # Text Area
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=15, font=("Arial", 12))
    text_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    text_area.bind("<KeyRelease>", on_text_change)

    # Prediction Display
    prediction_label = tk.Label(root, text="Prediction: (Initializing...)", font=("Arial", 12, "italic"), pady=5)
    prediction_label.pack()

    # Status Label for model loading
    status_label = tk.Label(root, text="Initializing model...", font=("Arial", 10), relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=2, padx=2)

    # Start model loading in a separate thread so GUI is responsive
    # This thread will update status_label via root.after
    model_load_thread = threading.Thread(target=load_model, daemon=True)
    model_load_thread.start()

    print("GUI setup complete. Starting mainloop.")
    root.mainloop()

# --- Main ---
if __name__ == "__main__":
    print("Script starting...")
    print(f"Using device: {DEVICE}")
    print(f"Model to load: {MODEL_NAME}")
    print(f"Max input tokens for prediction: {MAX_INPUT_TOKENS_FOR_PREDICTION}")
    print(f"Max new tokens to generate: {MAX_NEW_TOKENS_TO_GENERATE}")
    setup_gui()
