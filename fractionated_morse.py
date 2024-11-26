# Morse Code Mapping
MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 
    'Z': '--..'
} 

# Fractionation Table
FRACTIONATION_TABLE = {
    '...': 'A', '..-': 'B', '.-.': 'C', '.--': 'D', 
    '-..': 'E', '--.': 'F', '---': 'G', '-.-': 'H', 
    '.--': 'I', '-..': 'J', '--.': 'K', '---': 'L'
}

def text_to_morse(text):
    """Convert text to Morse code."""
    morse = ''
    for char in text.upper():
        if char in MORSE_CODE:
            morse += MORSE_CODE[char]
    return morse

def morse_to_fractionated(morse):
    """Convert Morse code to fractionated ciphertext."""
    fractionated = ''
    for i in range(0, len(morse), 3):
        chunk = morse[i:i + 3]
        if len(chunk) < 3:
            chunk = chunk.ljust(3, '.')
        fractionated += FRACTIONATION_TABLE.get(chunk, '?')  # '?' for unmatched patterns
    return fractionated

def fractionated_morse_encrypt(plaintext):
    """Encrypt plaintext using Fractionated Morse cipher."""
    morse = text_to_morse(plaintext)
    ciphertext = morse_to_fractionated(morse)
    return ciphertext
