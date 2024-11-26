import string
import random

def generate_grandpre_table():
    """Generate a complete substitution table for Grandpré cipher."""
    alphabet = string.ascii_uppercase
    digraphs = [a + b for a in alphabet for b in alphabet]  # All possible digraphs (AA to ZZ)
    symbols = list(alphabet + string.digits + "!@#$%^&*()-_+=<>?/")  # Use letters, numbers, and special characters
    
    # Ensure there are enough symbols (repeat the symbols if needed)
    while len(symbols) < len(digraphs):
        symbols += symbols  # Repeat the symbols until it covers all digraphs
    
    # Shuffle the symbols to randomize the substitution
    random.shuffle(symbols)
    
    # Generate the substitution table mapping each digraph to a symbol
    return dict(zip(digraphs, symbols[:len(digraphs)]))

# Substitution table for Grandpré cipher
GRANDPRE_TABLE = generate_grandpre_table()

def preprocess_text(text):
    """Preprocess text: remove non-alphabetic characters and convert to uppercase."""
    return ''.join(filter(str.isalpha, text.upper()))

def pair_text(text):
    """Pair text into digraphs, adding padding if needed."""
    if len(text) % 2 != 0:
        text += "X"  # Padding with 'X'
    return [text[i:i + 2] for i in range(0, len(text), 2)]

def grandpre_encrypt(plaintext):
    """Encrypt plaintext using the Grandpré cipher."""
    preprocessed_text = preprocess_text(plaintext)
    digraphs = pair_text(preprocessed_text)
    return ''.join(GRANDPRE_TABLE.get(pair, '?') for pair in digraphs)

# # Test the implementation
# if __name__ == "__main__":
#     plaintext = "THIS IS A SAMPLE PLAINTEXT FOR GRANDPRE CIPHER"
#     ciphertext = grandpre_encrypt(plaintext)
#     print(f"Plaintext: {plaintext}")
#     print(f"Ciphertext: {ciphertext}")
